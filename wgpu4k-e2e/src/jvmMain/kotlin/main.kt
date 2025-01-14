import io.github.oshai.kotlinlogging.KotlinLogging
import io.ygdrasil.webgpu.examples.headless.captureScene
import kotlinx.coroutines.runBlocking

private val logger = KotlinLogging.logger {}


fun main() {
    println("Session type ${System.getProperty("XDG_SESSION_TYPE")}")
    runBlocking {
        try {
            captureScene()
        } catch (error: Throwable) {
            logger.error(error) { "Failed to capture scene" }
        }
    }
}