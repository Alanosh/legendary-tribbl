package com.xai.memecoin

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import kotlinx.coroutines.*
import okhttp3.*
import okio.ByteString
import com.solana.api.*
import com.solana.core.PublicKey
import java.io.File
import java.time.Instant
import java.time.temporal.ChronoUnit
import java.util.logging.FileHandler
import java.util.logging.Logger
import java.util.logging.SimpleFormatter
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt

object WebSocketFetcher {
    private const val SOLANA_WS_URL = "wss://api.mainnet-beta.solana.com"
    private const val SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"
    private const val DATA_DIR = "memecoin_data"
    private const val OUTPUT_JSON = "$DATA_DIR/realtime_data.json"
    private const val ERROR_LOG = "$DATA_DIR/fetcher_errors.log"
    private const val INTERVAL_SECONDS = 10L
    private const val INTERVALS_PER_COIN = 361
    private const val LAST_VALID_JSON = "$DATA_DIR/last_valid_data.json"
    private val FEATURES = listOf(
        "Price_Surge", "Volume_SOL", "Insider_Ratio", "Wallet_Diversity",
        "Liquidity_Volume", "Price_Volatility", "Price_Momentum",
        "Transaction_Frequency", "Whale_Volume", "Trade_Size_Variance",
        "Sniper_Bot_Activity", "Buy_Sell_Ratio", "Failed_Trade_Ratio", "TOTAL_NET_SOL"
    )

    private val logger: Logger = Logger.getLogger(WebSocketFetcher::class.java.name).apply {
        useParentHandlers = false
        val handler = FileHandler(ERROR_LOG, true)
        handler.formatter = SimpleFormatter()
        addHandler(handler)
    }

    private val objectMapper: ObjectMapper = jacksonObjectMapper()
    private val client = RpcClient(SOLANA_RPC_URL)
    private val dataCache = mutableMapOf<String, MutableList<Map<String, Any>>>()
    private val lastValidData = mutableMapOf<String, MutableList<Map<String, Any>>>()

    // Helper functions for statistical calculations
    private fun List<Double>.mean(): Double = if (isNotEmpty()) sum() / size else 0.0
    private fun List<Double>.std(): Double = if (size > 1) sqrt(map { (it - mean()).pow(2) }.mean()) else 0.0
    private fun List<Double>.variance(): Double = if (size > 1) map { (it - mean()).pow(2) }.mean() else 0.0

    private suspend fun fetchMemecoinAddresses(): List<String> {
        try {
            val signatures = client.getSignaturesForAddress(
                PublicKey("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"),
                mapOf("limit" to 100)
            )
            val addresses = mutableSetOf<String>()
            val currentTime = Instant.now().epochSecond

            signatures.forEach { sig ->
                val tx = client.getTransaction(sig.signature, mapOf("encoding" to "jsonParsed", "commitment" to "confirmed"))
                val blockTime = tx?.blockTime?.toLong() ?: currentTime
                val meta = tx?.meta ?: return@forEach
                if (meta.innerInstructions.isNotEmpty()) {
                    meta.innerInstructions.forEach { instr ->
                        instr.instructions.forEach { inner ->
                            if (inner.programId.toString() == "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA") {
                                inner.accounts.forEach { key ->
                                    val addr = key.toString()
                                    val txCount = signatures.count { s ->
                                        s.accountKeys.any { it.toString() == addr }
                                    }
                                    if (txCount > 25 && blockTime > currentTime - 86400) {
                                        addresses.add(addr)
                                        if (addresses.size >= 10) return addresses.take(10)
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return addresses.take(10)
        } catch (e: Exception) {
            logger.severe("Fetch memecoin addresses error: ${e.message}")
            return emptyList()
        }
    }

    private suspend fun fetchTransactionData(address: String): Map<String, Any> {
        try {
            val pubkey = PublicKey(address)
            val signatures = client.getSignaturesForAddress(pubkey, mapOf("limit" to 100))
            if (signatures.isEmpty()) {
                throw IllegalStateException("No recent transactions for $address")
            }

            val txData = mutableListOf<Map<String, Any>>()
            signatures.forEach { sig ->
                val tx = client.getTransaction(sig.signature, mapOf("encoding" to "jsonParsed", "commitment" to "confirmed"))
                val meta = tx?.meta ?: return@forEach
                val blockTime = tx.blockTime?.toLong() ?: Instant.now().epochSecond
                val price = if (meta.preBalances.isNotEmpty() && meta.preBalances[0] != 0L) {
                    meta.postBalances.getOrNull(0)?.toDouble()?.div(meta.preBalances[0])?.minus(1) ?: 0.0
                } else 0.0
                val volume = meta.postBalances.getOrNull(0)?.toDouble()?.div(1e9) ?: 0.0
                txData.add(
                    mapOf(
                        "timestamp" to blockTime,
                        "price" to price,
                        "volume" to volume,
                        "accounts" to meta.accountKeys.size,
                        "status" to (if (meta.err != null) "failed" else "success")
                    )
                )
            }
            return mapOf("address" to address, "transactions" to txData)
        } catch (e: Exception) {
            logger.severe("Fetch error for $address: ${e.message}")
            return mapOf("address" to address, "transactions" to emptyList<Map<String, Any>>())
        }
    }

    private fun aggregateIntervalData(txData: Map<String, Any>, intervalStart: Instant): Map<String, Any>? {
        try {
            val address = txData["address"] as String
            val transactions = (txData["transactions"] as List<Map<String, Any>>).filter { tx ->
                val timestamp = (tx["timestamp"] as Long)
                val txTime = Instant.ofEpochSecond(timestamp)
                txTime >= intervalStart && txTime < intervalStart.plusSeconds(INTERVAL_SECONDS)
            }
            if (transactions.isEmpty()) return null

            val prices = transactions.map { it["price"] as Double }
            val volumes = transactions.map { it["volume"] as Double }
            val accounts = transactions.flatMap { List(it["accounts"] as Int) { it } }.toSet()

            // Compute Features
            val meanPrice = prices.mean()
            val sumVolume = volumes.sum()
            val meanVolume = volumes.mean()
            val stdVolume = volumes.std()
            val data = mutableMapOf<String, Any>(
                "Contract_Address" to address,
                "Interval" to (intervalStart.epochSecond / INTERVAL_SECONDS).toInt(),
                "Price_Surge" to meanPrice,
                "Volume_SOL" to sumVolume,
                "Insider_Ratio" to volumes.count { it > meanVolume + 2 * stdVolume }.toDouble() / max(volumes.size, 1),
                "Wallet_Diversity" to accounts.size.toDouble() / max(transactions.size, 1),
                "Liquidity_Volume" to sumVolume * 0.8,
                "Price_Volatility" to prices.std(),
                "Price_Momentum" to if (prices.isNotEmpty() && prices.first() != 0.0) {
                    (prices.last() - prices.first()) / prices.first()
                } else 0.0,
                "Transaction_Frequency" to transactions.size.toDouble() / INTERVAL_SECONDS,
                "Whale_Volume" to volumes.filter { it > meanVolume + 2 * stdVolume }.sum(),
                "Trade_Size_Variance" to volumes.variance(),
                "Sniper_Bot_Activity" to transactions.count {
                    (it["timestamp"] as Long) < intervalStart.epochSecond + 2
                }.toDouble() / max(transactions.size, 1),
                "Buy_Sell_Ratio" to max(transactions.count { (it["price"] as Double) > 0 }.toDouble() /
                        max(transactions.count { (it["price"] as Double) < 0 }, 1), 1.0),
                "Failed_Trade_Ratio" to transactions.count {
                    it["status"] == "failed"
                }.toDouble() / max(transactions.size, 1),
                "TOTAL_NET_SOL" to sumVolume
            )

            // Pre-Filtering
            when {
                data["Sniper_Bot_Activity"] as Double > 0.8 && (data["Interval"] as Int) < 90 ->
                    data["Priority"] = "High"
                (data["Liquidity_Volume"] as Double) < 0.4 * meanVolume && (data["Insider_Ratio"] as Double) > 0.75 ->
                    data["Priority"] = "Low"
                else -> data["Priority"] = "Medium"
            }

            return data
        } catch (e: Exception) {
            logger.severe("Aggregation error for ${txData["address"]}: ${e.message}")
            return null
        }
    }

    private suspend fun saveIntervalData() {
        try {
            val intervalStart = Instant.now().truncatedTo(ChronoUnit.SECONDS)
                .minusSeconds(INTERVAL_SECONDS)
            val outputData = mutableListOf<Map<String, Any>>()

            // Fetch memecoin addresses if cache is empty
            if (dataCache.isEmpty()) {
                val memecoinAddresses = fetchMemecoinAddresses()
                memecoinAddresses.forEach { addr ->
                    dataCache[addr] = mutableListOf()
                    lastValidData[addr] = mutableListOf()
                }
            }

            for (address in dataCache.keys) {
                val txData = fetchTransactionData(address)
                val intervalData = aggregateIntervalData(txData, intervalStart)
                if (intervalData != null) {
                    dataCache[address]!!.add(intervalData)
                    if (dataCache[address]!!.size > INTERVALS_PER_COIN) {
                        dataCache[address] = dataCache[address]!!.takeLast(INTERVALS_PER_COIN).toMutableList()
                    }
                    outputData.add(intervalData)
                    lastValidData[address] = dataCache[address]!!.toMutableList()
                }
            }

            if (outputData.isNotEmpty()) {
                File(OUTPUT_JSON).writeText(objectMapper.writeValueAsString(outputData))
                File(LAST_VALID_JSON).writeText(objectMapper.writeValueAsString(lastValidData))
            }
        } catch (e: Exception) {
            logger.severe("Save error: ${e.message}")
        }
    }

    private suspend fun websocketLoop() {
        var retryDelay = 1L
        while (true) {
            try {
                val webSocketClient = OkHttpClient()
                val request = Request.Builder().url(SOLANA_WS_URL).build()
                val webSocket = webSocketClient.newWebSocket(request, object : WebSocketListener() {
                    override fun onOpen(webSocket: WebSocket, response: Response) {
                        dataCache.keys.forEach { addr ->
                            webSocket.send(
                                """{"jsonrpc": "2.0","id": 1,"method": "accountSubscribe","params": ["$addr", {"encoding": "jsonParsed"}]}"""
                            )
                        }
                    }

                    override fun onMessage(webSocket: WebSocket, text: String) {
                        // Placeholder: Actual data processed in saveIntervalData
                    }

                    override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                        // Ignore binary messages
                    }

                    override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                        webSocket.close(1000, null)
                    }

                    override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                        logger.severe("WebSocket error: ${t.message}")
                    }
                })

                while (true) {
                    saveIntervalData()
                    delay(INTERVAL_SECONDS * 1000)
                }
            } catch (e: Exception) {
                logger.severe("WebSocket error: ${e.message}")
                retryDelay = minOf(retryDelay * 2, 16)
                delay(retryDelay * 1000)
            }
        }
    }

    @JvmStatic
    fun main(args: Array<String>) {
        try {
            File(DATA_DIR).mkdirs()
            runBlocking {
                websocketLoop()
            }
        } catch (e: Exception) {
            logger.severe("Main error: ${e.message}")
            throw e
        }
    }
}
