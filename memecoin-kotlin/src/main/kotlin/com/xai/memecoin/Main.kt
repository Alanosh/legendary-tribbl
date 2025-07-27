package com.xai.memecoin

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import smile.classification.GradientTreeBoost
import smile.clustering.KMeans
import smile.data.DataFrame
import smile.data.formula.Formula
import smile.data.vector.DoubleVector
import smile.io.Read
import smile.math.MathEx
import smile.math.distance.EuclideanDistance
import smile.validation.ClassificationMetrics
import java.io.File
import java.io.FileWriter
import java.util.logging.FileHandler
import java.util.logging.Logger
import java.util.logging.SimpleFormatter
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

object Main {
    private const val DATA_DIR = "memecoin_data"
    private const val PUMP_FILE = "$DATA_DIR/pumps_combined.csv"
    private const val DUD_FILE = "$DATA_DIR/duds_combined.csv"
    private const val INTERVALS_PER_COIN = 361
    private const val CONFIDENCE_THRESHOLD_PUMP = 0.9
    private const val CONFIDENCE_THRESHOLD_DUD = 0.8
    private const val CACHE_FILE = "cache.json"
    private const val ERROR_LOG = "model_errors.log"
    private const val PREDICTIONS_FILE = "predictions.txt"
    private const val MODEL_MAIN = "gradient_boost_main.model"
    private const val MODEL_HONEYPOT = "gradient_boost_honeypot.model"
    private const val ANOMALY_MODEL = "anomaly_detector.model"

    private val logger: Logger = Logger.getLogger(Main::class.java.name).apply {
        val handler = FileHandler(ERROR_LOG, true)
        handler.formatter = SimpleFormatter()
        addHandler(handler)
    }

    private val objectMapper: ObjectMapper = jacksonObjectMapper()

    private val features = listOf(
        "Price_Surge", "Volume_SOL", "Insider_Ratio", "Wallet_Diversity",
        "Liquidity_Volume", "Price_Volatility", "Price_Momentum",
        "Transaction_Frequency", "Whale_Volume", "Trade_Size_Variance",
        "Sniper_Bot_Activity", "Buy_Sell_Ratio", "Failed_Trade_Ratio", "TOTAL_NET_SOL"
    )

    // Helper functions for preprocessing
    private fun List<Double>.median(): Double {
        val sorted = sorted()
        val mid = sorted.size / 2
        return if (sorted.size % 2 == 0) (sorted[mid - 1] + sorted[mid]) / 2 else sorted[mid]
    }

    private fun List<Double>.quantile(q: Double): Double {
        val sorted = sorted()
        val pos = (sorted.size - 1) * q
        val base = pos.toInt()
        val rest = pos - base
        return if (base + 1 < sorted.size) sorted[base] * (1 - rest) + sorted[base + 1] * rest else sorted[base]
    }

    private fun List<Double>.mean(): Double = sum() / size
    private fun List<Double>.std(): Double = sqrt(map { (it - mean()).pow(2) }.mean())

    private fun zscore(values: List<Double>): List<Double> {
        val mean = values.mean()
        val std = values.std()
        return values.map { (it - mean) / (std + 1e-6) }
    }

    private fun correlation(x: List<Double>, y: List<Double>): Double {
        val meanX = x.mean()
        val meanY = y.mean()
        val cov = x.zip(y).map { (a, b) -> (a - meanX) * (b - meanY) }.sum() / x.size
        val stdX = x.std()
        val stdY = y.std()
        return if (stdX == 0.0 || stdY == 0.0) 0.0 else cov / (stdX * stdY)
    }

    private fun ks2samp(data1: List<Double>, data2: List<Double>): Pair<Double, Double> {
        val sorted1 = data1.sorted()
        val sorted2 = data2.sorted()
        val n1 = sorted1.size
        val n2 = sorted2.size
        var i1 = 0
        var i2 = 0
        var d = 0.0
        while (i1 < n1 && i2 < n2) {
            val v1 = sorted1[i1]
            val v2 = sorted2[i2]
            if (v1 <= v2) i1++ else i2++
            val cdf1 = i1.toDouble() / n1
            val cdf2 = i2.toDouble() / n2
            d = max(d, abs(cdf1 - cdf2))
        }
        val pValue = 1.0 - MathEx.ksPValue(d, n1.toLong(), n2.toLong())
        return d to pValue
    }

    // 1. Data Preprocessing and Feature Engineering
    private fun loadAndMergeData(): DataFrame {
        try {
            val pumps = Read.csv(PUMP_FILE).apply {
                add(DoubleVector.of("Is_Pump", DoubleArray(size()) { 1.0 }))
            }
            val duds = Read.csv(DUD_FILE).apply {
                add(DoubleVector.of("Is_Pump", DoubleArray(size()) { 0.0 }))
            }
            var df = DataFrame.of(pumps, duds)
            df = df.orderBy("Contract_Address", "Interval")
            val counts = df.groupBy("Contract_Address").mapValues { it.value.size() }
            if (counts.any { it.value != INTERVALS_PER_COIN }) {
                throw IllegalArgumentException("Invalid interval count per Contract_Address")
            }
            return df
        } catch (e: Exception) {
            logger.severe("Load error: ${e.message}")
            throw e
        }
    }

    private fun preprocessData(df: DataFrame, isTraining: Boolean = true): DataFrame {
        var newDf = df.select(features + "Contract_Address" + "Interval" + if (isTraining) "Is_Pump" else "")

        // Dynamic RobustScaler per Contract_Address
        for (addr in newDf["Contract_Address"].toStringArray().distinct()) {
            val mask = newDf.filter { it.getString("Contract_Address") == addr }
            val data = features.map { col -> mask[col].toDoubleArray() }
            val scaled = data.map { col ->
                val median = col.median()
                val iqr = col.quantile(0.75) - col.quantile(0.25)
                col.map { (it - median) / (iqr + 1e-6) }.toDoubleArray()
            }
            features.forEachIndexed { i, col ->
                mask.update(col, scaled[i])
            }
        }

        // Adaptive Outlier Clipping
        val volatility = newDf["Price_Volatility"].toDoubleArray().toList().rollingMean(INTERVALS_PER_COIN)
        val quantile = if (volatility.mean() > volatility.std()) 0.005 else 0.01
        for (col in listOf("Volume_SOL", "Price_Volatility", "Whale_Volume", "Transaction_Frequency")) {
            val grouped = newDf.groupBy("Contract_Address")
            for ((_, group) in grouped) {
                val values = group[col].toDoubleArray()
                val lower = values.quantile(quantile)
                val upper = values.quantile(1 - quantile)
                group.update(col, values.map { min(max(it, lower), upper) }.toDoubleArray())
            }
        }

        // Adaptive EMA Smoothing
        for (col in listOf("Price_Surge", "Price_Momentum", "Transaction_Frequency")) {
            val volatility = newDf["Price_Volatility"].toDoubleArray().toList().rollingStd(30)
            val alpha = volatility.map { if (it > volatility.mean()) 0.05 else 0.4 }
            val ema = newDf.groupBy("Contract_Address").mapValues { (addr, group) ->
                val values = group[col].toDoubleArray()
                values.mapIndexed { i, v ->
                    val prev = if (i == 0) v else group[i - 1]["${col}_EMA"]?.toDouble() ?: v
                    prev * (1 - alpha[i]) + v * alpha[i]
                }
            }
            newDf = newDf.add(DoubleVector.of("${col}_EMA", newDf["Contract_Address"].toStringArray().map { addr ->
                ema[addr]?.get(newDf.indexOf { it.getString("Contract_Address") == addr }) ?: newDf[col].toDoubleArray()[newDf.indexOf { it.getString("Contract_Address") == addr }]
            }.toDoubleArray()))
        }

        // Noise-Adjusted Features
        val volatilitySpike = newDf["Price_Volatility"].toDoubleArray().mapIndexed { i, vol ->
            val mean = newDf["Price_Volatility"].toDoubleArray().sliceArray(max(0, i - 30) until i + 1).mean()
            val std = newDf["Price_Volatility"].toDoubleArray().sliceArray(max(0, i - 30) until i + 1).std()
            if (vol > mean + 2.5 * std) newDf[i]["Insider_Ratio"].toDouble() else 0.0
        }
        val insiderClustering = newDf["Insider_Ratio"].toDoubleArray().zip(newDf["Wallet_Diversity"].toDoubleArray()).map { (insider, diversity) ->
            insider / (diversity + 1e-6)
        }
        val liquidityDecay = newDf.groupBy("Contract_Address").mapValues { (addr, group) ->
            val values = group["Liquidity_Volume"].toDoubleArray()
            values.mapIndexed { i, v -> if (i < 10) 0.0 else (v - values[i - 10]) / (values[i - 10] + 1e-6) }
        }
        val honeypotScore = newDf["Contract_Address"].toStringArray().mapIndexed { i, addr ->
            if (liquidityDecay[addr]!![newDf.indexOf { it.getString("Contract_Address") == addr }] < -0.6 &&
                newDf[i]["Insider_Ratio"].toDouble() > 0.75) 1.0 else 0.0
        }
        val crossCoinCorr = newDf.groupBy("Interval").mapValues { (interval, group) ->
            correlation(group["Price_Surge"].toDoubleArray().toList(), group["Volume_SOL"].toDoubleArray().toList())
        }
        val darkPoolSignal = newDf["Whale_Volume"].toDoubleArray().zip(newDf["Transaction_Frequency"].toDoubleArray()).mapIndexed { i, (whale, freq) ->
            val prev = if (i == 0) whale / (freq + 1e-6) else newDf[i - 1]["Dark_Pool_Signal"]?.toDouble() ?: whale / (freq + 1e-6)
            prev * 0.95 + (whale / (freq + 1e-6)) * 0.05
        }
        val darkPoolEnhanced = darkPoolSignal.zip(newDf["Interval"].toStringArray().map { crossCoinCorr[it] ?: 0.0 }).map { (signal, corr) -> signal * (1 - corr) }

        newDf = newDf.add(
            DoubleVector.of("Volatility_Spike_Indicator", volatilitySpike.toDoubleArray()),
            DoubleVector.of("Insider_Clustering_Ratio", insiderClustering.toDoubleArray()),
            DoubleVector.of("Liquidity_Decay_Metric", newDf["Contract_Address"].toStringArray().map { addr ->
                liquidityDecay[addr]!![newDf.indexOf { it.getString("Contract_Address") == addr }]
            }.toDoubleArray()),
            DoubleVector.of("Honeypot_Score", honeypotScore.toDoubleArray()),
            DoubleVector.of("Cross_Coin_Correlation", newDf["Interval"].toStringArray().map { crossCoinCorr[it] ?: 0.0 }.toDoubleArray()),
            DoubleVector.of("Dark_Pool_Signal", darkPoolSignal.toDoubleArray()),
            DoubleVector.of("Dark_Pool_Enhanced", darkPoolEnhanced.toDoubleArray())
        )

        // Temporal Feature Aggregation
        for (window in listOf(6, 30, 60)) {
            for (col in listOf("Price_Surge", "Volume_SOL", "Sniper_Bot_Activity")) {
                val maxVals = newDf.groupBy("Contract_Address").mapValues { (addr, group) ->
                    val values = group[col].toDoubleArray()
                    values.mapIndexed { i, _ ->
                        values.sliceArray(max(0, i - window + 1) until i + 1).maxOrNull() ?: values[i]
                    }
                }
                val meanVals = newDf.groupBy("Contract_Address").mapValues { (addr, group) ->
                    val values = group[col].toDoubleArray()
                    values.mapIndexed { i, _ ->
                        values.sliceArray(max(0, i - window + 1) until i + 1).mean()
                    }
                }
                val stdVals = newDf.groupBy("Contract_Address").mapValues { (addr, group) ->
                    val values = group[col].toDoubleArray()
                    values.mapIndexed { i, _ ->
                        values.sliceArray(max(0, i - window + 1) until i + 1).std()
                    }
                }
                newDf = newDf.add(
                    DoubleVector.of("${col}_Max_$window", newDf["Contract_Address"].toStringArray().map { addr ->
                        maxVals[addr]!![newDf.indexOf { it.getString("Contract_Address") == addr }]
                    }.toDoubleArray()),
                    DoubleVector.of("${col}_Mean_$window", newDf["Contract_Address"].toStringArray().map { addr ->
                        meanVals[addr]!![newDf.indexOf { it.getString("Contract_Address") == addr }]
                    }.toDoubleArray()),
                    DoubleVector.of("${col}_Std_$window", newDf["Contract_Address"].toStringArray().map { addr ->
                        stdVals[addr]!![newDf.indexOf { it.getString("Contract_Address") == addr }] ?: 0.0
                    }.toDoubleArray())
                )
            }
        }

        // Interaction Terms
        newDf = newDf.add(
            DoubleVector.of("Price_Surge_x_Insider_Ratio", newDf["Price_Surge"].toDoubleArray().zip(newDf["Insider_Ratio"].toDoubleArray()).map { (a, b) -> a * b }.toDoubleArray()),
            DoubleVector.of("Whale_Volume_div_Buy_Sell_Ratio", newDf["Whale_Volume"].toDoubleArray().zip(newDf["Buy_Sell_Ratio"].toDoubleArray()).map { (a, b) -> a / (b + 1e-6) }.toDoubleArray()),
            DoubleVector.of("Sniper_Bot_x_Liquidity", newDf["Sniper_Bot_Activity"].toDoubleArray().zip(newDf["Liquidity_Volume"].toDoubleArray()).map { (a, b) -> a * b }.toDoubleArray())
        )

        // Outlier Rejection with Z-Score
        for (col in listOf("Whale_Volume", "Transaction_Frequency")) {
            val grouped = newDf.groupBy("Contract_Address")
            for ((addr, group) in grouped) {
                val values = group[col].toDoubleArray()
                val zScores = zscore(values.toList().rollingMean(30))
                val median = values.toList().rollingMedian(30)
                group.update(col, values.mapIndexed { i, v -> if (abs(zScores[i]) > 3) median[i] else v }.toDoubleArray())
            }
        }

        return newDf
    }

    // 2. Model Training with Noise Injection
    private fun trainModel(df: DataFrame): Tuple<List<GradientTreeBoost>, GradientTreeBoost, GradientTreeBoost, IsolationForest, Map<String, Double>, KMeans> {
        try {
            val features = df.names().filter { it !in listOf("Contract_Address", "Interval", "Is_Pump") }
            val X = df.select(features)
            val y = df["Is_Pump"].toIntArray()

            // Synthetic Noise and Adversarial Augmentation
            var XNoisy = X.copy()
            for (col in features) {
                val noise = DoubleArray(X.size()) { Random.nextDouble(-0.35, 0.35) }
                XNoisy.update(col, XNoisy[col].toDoubleArray().mapIndexed { i, v -> v + noise[i] }.toDoubleArray())
            }
            XNoisy = XNoisy.sample((X.size() * 0.88).toInt())
            val advMask = List(XNoisy.size()) { Random.nextDouble() < 0.15 }
            for (i in advMask.indices) {
                if (advMask[i]) {
                    XNoisy.update("Volume_SOL", XNoisy["Volume_SOL"].toDoubleArray().mapIndexed { j, v -> if (j == i) v * 3 else v }.toDoubleArray())
                    XNoisy.update("Insider_Ratio", XNoisy["Insider_Ratio"].toDoubleArray().mapIndexed { j, v -> if (j == i) v * 1.5 else v }.toDoubleArray())
                }
            }

            // Temporal Boosting
            var sampleWeights = DoubleArray(df.size()) { 1.0 }
            df["Interval"].toIntArray().forEachIndexed { i, interval ->
                if (interval >= INTERVALS_PER_COIN - 90) sampleWeights[i] *= 1.7
            }

            // Noise-Adaptive Loss
            val volatility = df["Price_Volatility"].toDoubleArray()
            val noiseMask = volatility.map { it > volatility.mean() + 2.5 * volatility.std() }
            sampleWeights = sampleWeights.mapIndexed { i, w -> if (noiseMask[i]) w * 0.4 else w }.toDoubleArray()

            // Archetype Fingerprinting
            val reducer = smile.manifold.UMAP.of(X.select("Price_Surge", "Insider_Ratio", "Liquidity_Volume").toArray(), 3)
            val clusters = reducer.coordinates
            val kmeans = KMeans.fit(clusters, 8)
            val clusterLabels = kmeans.predict(clusters)

            // Cluster-Based Ensemble
            val models = mutableListOf<GradientTreeBoost>()
            for (cluster in 0 until 8) {
                val mask = clusterLabels.map { it == cluster }
                val XCluster = X.filter { mask[X.indexOf(it)] }
                val yCluster = y.filterIndexed { i, _ -> mask[i] }.toIntArray()
                val weightsCluster = sampleWeights.filterIndexed { i, _ -> mask[i] }.toDoubleArray()
                val model = GradientTreeBoost.fit(Formula.lhs("Is_Pump"), XCluster, yCluster,
                    ntrees = 300, maxDepth = 7, shrinkage = 0.08, samplingRate = 0.67
                )
                val tscv = (0 until 6).map { i ->
                    val trainEnd = i * (XCluster.size() / 6)
                    val valStart = trainEnd
                    val valEnd = min((i + 1) * (XCluster.size() / 6), XCluster.size())
                    (0 until trainEnd).toList() to (valStart until valEnd).toList()
                }
                for ((trainIdx, valIdx) in tscv) {
                    val XTrain = XCluster.filter { trainIdx.contains(XCluster.indexOf(it)) }
                    val yTrain = trainIdx.map { yCluster[it] }.toIntArray()
                    val XVal = XCluster.filter { valIdx.contains(XCluster.indexOf(it)) }
                    val yVal = valIdx.map { yCluster[it] }.toIntArray()
                    val tempModel = GradientTreeBoost.fit(Formula.lhs("Is_Pump"), XTrain, yTrain,
                        ntrees = 300, maxDepth = 7, shrinkage = 0.08, samplingRate = 0.67
                    )
                    val metrics = ClassificationMetrics.of(yVal, tempModel.predict(XVal))
                    println("Cluster $cluster: Accuracy=${metrics.accuracy}")
                }
                models.add(model)
            }

            // Main Model
            val mainModel = GradientTreeBoost.fit(Formula.lhs("Is_Pump"), XNoisy, y,
                ntrees = 300, maxDepth = 7, shrinkage = 0.08, samplingRate = 0.67
            )

            // Honeypot Ensemble
            val honeypotModel = GradientTreeBoost.fit(Formula.lhs("Is_Pump"),
                X.select("Honeypot_Score", "Liquidity_Decay_Metric", "Insider_Clustering_Ratio"), y,
                ntrees = 200, maxDepth = 5, shrinkage = 0.1, samplingRate = 0.67
            )

            // Anomaly Detection
            val anomalyDetector = smile.clustering.IsolationForest.fit(X.toArray(), ntrees = 200, psi = 0.05)

            // Feature Importance Reweighting
            val importance = mainModel.importance()
            val featureWeights = features.zip(importance).toMap()
            val noisyFeatures = features.filter { df[it].toDoubleArray().std() > 2 * df[it].toDoubleArray().std().mean() }
            val finalWeights = featureWeights.mapValues { (k, v) -> if (k in noisyFeatures) v * 0.5 else v }
            val XWeighted = X.copy()
            for (col in features) {
                XWeighted.update(col, XWeighted[col].toDoubleArray().mapIndexed { i, v -> v * finalWeights[col]!! }.toDoubleArray())
            }

            // Save Archetype Profiles
            val pumpProfile = X.filter { df[X.indexOf(it)]["Is_Pump"].toInt() == 1 }.mean()
            val dudProfile = X.filter { df[X.indexOf(it)]["Is_Pump"].toInt() == 0 }.mean()
            val cache = mapOf(
                "pump_profile" to features.zip(pumpProfile.toDoubleArray()).toMap(),
                "dud_profile" to features.zip(dudProfile.toDoubleArray()).toMap(),
                "centroids" to kmeans.centroids.map { it.toList() },
                "price_surge_mean" to X["Price_Surge"].toDoubleArray().mean()
            )
            File(CACHE_FILE).writeText(objectMapper.writeValueAsString(cache))

            return Tuple(models, mainModel, honeypotModel, anomalyDetector, finalWeights, kmeans)
        } catch (e: Exception) {
            logger.severe("Train error: ${e.message}")
            throw e
        }
    }

    // 3. Generic Data Processing
    private fun processData(
        dataChunk: List<Map<String, Any>>,
        models: List<GradientTreeBoost>,
        mainModel: GradientTreeBoost,
        honeypotModel: GradientTreeBoost,
        anomalyDetector: smile.clustering.IsolationForest,
        featureWeights: Map<String, Double>,
        kmeans: KMeans
    ): List<Map<String, Any>> {
        try {
            val df = DataFrame.of(dataChunk.map { row ->
                features.associateWith { row[it]?.toString()?.toDouble() ?: throw IllegalArgumentException("Missing $it") } +
                    mapOf("Contract_Address" to row["Contract_Address"].toString(), "Interval" to row["Interval"].toString().toInt())
            })
            if (!features.all { it in df.names() } || !listOf("Contract_Address", "Interval").all { it in df.names() }) {
                throw IllegalArgumentException("Invalid input data format")
            }

            val processedDf = preprocessData(df, isTraining = false)
            val features = processedDf.names().filter { it !in listOf("Contract_Address", "Interval") }
            val X = processedDf.select(features)

            // Dynamic Thresholding
            val volatility = X["Price_Volatility"].toDoubleArray().mean()
            val volMean = X["Price_Volatility"].toDoubleArray().toList().rollingMean(INTERVALS_PER_COIN * 24)
            val pumpThreshold = if (volatility > volMean) 0.85 else 0.9
            val dudThreshold = if (volatility > volMean) 0.75 else 0.8

            // Ensemble Predictions
            val probs = models.map { model ->
                model.predictProba(X).map { it[1] }
            }.fold(DoubleArray(X.size()) { 0.0 }) { acc, p ->
                acc.mapIndexed { i, v -> v + p[i] / models.size }.toDoubleArray()
            }
            val honeypotProbs = honeypotModel.predictProba(X.select("Honeypot_Score", "Liquidity_Decay_Metric", "Insider_Clustering_Ratio")).map { it[1] }
            val finalProbs = probs.zip(honeypotProbs).map { (p, hp) -> 0.7 * p + 0.3 * (1 - hp) }
            val preds = finalProbs.map { if (it >= pumpThreshold) 1 else 0 }.toIntArray()

            // Uncertainty (simplified variance across models)
            val treePreds = models.map { it.predictProba(X).map { p -> p[1] } }
            val confidenceIntervals = finalProbs.mapIndexed { i, p ->
                val std = treePreds.map { it[i] }.std()
                "${"%.2f".format(p)}±${"%.2f".format(std)}"
            }

            // Archetype Similarity
            val cache = objectMapper.readValue(File(CACHE_FILE), Map::class.java) as Map<String, Any>
            val pumpProfile = (cache["pump_profile"] as Map<String, Double>).values.toDoubleArray()
            val dudProfile = (cache["dud_profile"] as Map<String, Double>).values.toDoubleArray()
            val centroids = (cache["centroids"] as List<List<Double>>).map { it.toDoubleArray() }
            val similarityScores = preds.mapIndexed { i, p ->
                1 - EuclideanDistance().d(X[i].toDoubleArray(), if (p == 1) pumpProfile else dudProfile)
            }
            val clusterDistances = X.toArray().map { x -> centroids.map { c -> EuclideanDistance().d(x, c) }.minOrNull()!! }

            // Temporal Amplification
            val amplifiedX = X.copy()
            amplifiedX["Interval"].toIntArray().forEachIndexed { i, interval ->
                if (interval >= INTERVALS_PER_COIN - 90) {
                    amplifiedX.update("Price_Momentum", amplifiedX["Price_Momentum"].toDoubleArray().mapIndexed { j, v -> if (j == i) v * 1.7 else v }.toDoubleArray())
                    amplifiedX.update("Sniper_Bot_Activity", amplifiedX["Sniper_Bot_Activity"].toDoubleArray().mapIndexed { j, v -> if (j == i) v * 1.7 else v }.toDoubleArray())
                }
            }

            // Anomalies
            val anomalies = anomalyDetector.predict(X.toArray()).map { it == -1 }

            // Output Results
            val results = mutableListOf<Map<String, Any>>()
            for (i in preds.indices) {
                if ((preds[i] == 1 && finalProbs[i] >= pumpThreshold) || (preds[i] == 0 && finalProbs[i] <= 1 - dudThreshold)) {
                    val topFeatures = features.map { it to featureWeights.getOrDefault(it, 0.0) }
                        .sortedByDescending { it.second }.take(3).map { it.first }
                    val result = mapOf(
                        "Contract_Address" to processedDf[i]["Contract_Address"].toString(),
                        "Predicted_Class" to if (preds[i] == 1) "Pump" else "Dud",
                        "Confidence_Score" to finalProbs[i],
                        "Confidence_Interval" to confidenceIntervals[i],
                        "Similarity_Score" to similarityScores[i],
                        "Cluster_Distance" to clusterDistances[i],
                        "Key_Features" to topFeatures,
                        "Is_Anomaly" to anomalies[i]
                    )
                    results.add(result)
                    FileWriter(PREDICTIONS_FILE, true).use { it.write("${objectMapper.writeValueAsString(result)}\n") }
                    if (preds[i] == 1 && finalProbs[i] >= pumpThreshold) {
                        println("ALERT: Pump detected! $result")
                    }
                }
            }
            return results
        } catch (e: Exception) {
            logger.severe("Process error: ${e.message}")
            return emptyList()
        }
    }

    // 4. Model Adaptation and Monitoring
    private fun updateModelIncrementally(
        dataChunk: List<Map<String, Any>>,
        models: List<GradientTreeBoost>,
        mainModel: GradientTreeBoost,
        honeypotModel: GradientTreeBoost,
        kmeans: KMeans
    ): Triple<List<GradientTreeBoost>, GradientTreeBoost, GradientTreeBoost> {
        try {
            val df = DataFrame.of(dataChunk.map { row ->
                (features + listOf("Contract_Address", "Interval", "Is_Pump")).associateWith {
                    row[it]?.toString()?.toDouble() ?: if (it == "Is_Pump") 0.0 else throw IllegalArgumentException("Missing $it")
                }
            })
            val processedDf = preprocessData(df, isTraining = false)
            val features = processedDf.names().filter { it !in listOf("Contract_Address", "Interval", "Is_Pump") }
            val X = processedDf.select(features)
            val y = processedDf["Is_Pump"].toIntArray()

            var sampleWeights = DoubleArray(df.size()) { 1.0 }
            processedDf["Interval"].toIntArray().forEachIndexed { i, interval ->
                if (interval >= INTERVALS_PER_COIN - 90) sampleWeights[i] *= 1.7
            }
            val volatility = processedDf["Price_Volatility"].toDoubleArray()
            val noiseMask = volatility.map { it > volatility.mean() + 2.5 * volatility.std() }
            sampleWeights = sampleWeights.mapIndexed { i, w -> if (noiseMask[i]) w * 0.4 else w }.toDoubleArray()

            val newModels = models.map { model ->
                GradientTreeBoost.fit(Formula.lhs("Is_Pump"), X, y, ntrees = 300, maxDepth = 7, shrinkage = 0.008, samplingRate = 0.67)
            }
            val newMainModel = GradientTreeBoost.fit(Formula.lhs("Is_Pump"), X, y, ntrees = 300, maxDepth = 7, shrinkage = 0.008, samplingRate = 0.67)
            val newHoneypotModel = GradientTreeBoost.fit(Formula.lhs("Is_Pump"),
                X.select("Honeypot_Score", "Liquidity_Decay_Metric", "Insider_Clustering_Ratio"), y,
                ntrees = 200, maxDepth = 5, shrinkage = 0.008, samplingRate = 0.67
            )

            // Drift Detection
            val cache = objectMapper.readValue(File(CACHE_FILE), Map::class.java) as Map<String, Any>
            val historicalDist = cache["price_surge_mean"] as? Double ?: X["Price_Surge"].toDoubleArray().mean()
            val (_, pValue) = ks2samp(
                X["Price_Surge"].toDoubleArray().toList(),
                List(X.size()) { Random.nextDouble(historicalDist - X["Price_Surge"].toDoubleArray().std(), historicalDist + X["Price_Surge"].toDoubleArray().std()) }
            )
            if (pValue < 0.03) {
                println("Drift detected, retraining...")
                val allDf = DataFrame.of(loadAndMergeData(), processedDf)
                return trainModel(allDf).let { (m, mm, hm, _, _, k) -> Triple(m, mm, hm) }
            }
            val newCache = cache.toMutableMap().apply { put("price_surge_mean", X["Price_Surge"].toDoubleArray().mean()) }
            File(CACHE_FILE).writeText(objectMapper.writeValueAsString(newCache))

            return Triple(newModels, newMainModel, newHoneypotModel)
        } catch (e: Exception) {
            logger.severe("Update error: ${e.message}")
            return Triple(models, mainModel, honeypotModel)
        }
    }

    // 5. Main Pipeline
    @JvmStatic
    fun main(args: Array<String>) {
        try {
            // Load and Train
            val df = loadAndMergeData()
            val (models, mainModel, honeypotModel, anomalyDetector, featureWeights, kmeans) = trainModel(df)

            // Save Models
            File(MODEL_MAIN).writeBytes(mainModel.toBytes())
            models.forEachIndexed { i, model -> File("gradient_boost_cluster_$i.model").writeBytes(model.toBytes()) }
            File(MODEL_HONEYPOT).writeBytes(honeypotModel.toBytes())
            File(ANOMALY_MODEL).writeBytes(anomalyDetector.toBytes())

            // Process Input Data
            var lastMtime = 0L
            while (true) {
                try {
                    val file = File("realtime_data.json")
                    if (file.exists()) {
                        val mtime = file.lastModified()
                        if (mtime > lastMtime) {
                            val dataChunk = objectMapper.readValue(file, List::class.java) as List<Map<String, Any>>
                            val results = processData(dataChunk, models, mainModel, honeypotModel, anomalyDetector, featureWeights, kmeans)
                            val (updatedModels, updatedMainModel, updatedHoneypotModel) = updateModelIncrementally(dataChunk, models, mainModel, honeypotModel, kmeans)
                            models.clear()
                            models.addAll(updatedModels)
                            lastMtime = mtime
                        }
                    }
                    Thread.sleep(1000) // Poll every second
                } catch (e: Exception) {
                    logger.severe("Pipeline loop error: ${e.message}")
                }
            }
        } catch (e: Exception) {
            logger.severe("Pipeline error: ${e.message}")
            throw e
        }
    }

    private fun List<Double>.rollingMean(window: Int): Double {
        return subList(max(0, size - window), size).mean()
    }

    private fun List<Double>.rollingStd(window: Int): Double {
        return subList(max(0, size - window), size).std()
    }

    private fun List<Double>.rollingMedian(window: Int): Double {
        return subList(max(0, size - window), size).median()
    }

    data class Tuple<A, B, C, D, E, F>(
        val first: A,
        val second: B,
        val third: C,
        val fourth: D,
        val fifth: E,
        val sixth: F
    ) {
        constructor(first: A, second: B, third: C) : this(first, second, third, null as D, null as E, null as F)
    }
}
