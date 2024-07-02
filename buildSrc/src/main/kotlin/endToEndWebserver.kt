import com.microsoft.playwright.*
import io.ktor.server.engine.*
import io.ktor.server.http.content.*
import io.ktor.server.netty.*
import io.ktor.server.routing.*
import org.gradle.api.logging.Logger
import java.io.File
import java.util.*


val scenes = listOf(
    "HelloTriangleScene" to listOf(0),
    "HelloTriangleMSAAScene" to listOf(0),
    "HelloTriangleRotatingScene" to listOf(0, 10, 50, 100),
    "RotatingCubeScene" to listOf(0, 10, 50, 100),
    "TwoCubesScene" to listOf(0, 10, 50, 100),
    "CubemapScene" to listOf(0, 10, 50, 100),
    "InstancedCubeScene" to listOf(0, 10, 50, 100),
    "TexturedCubeScene" to listOf(0, 10, 50, 100),
)

fun endToEndWebserver(basePath: File): NettyApplicationEngine {
    val pagePath = basePath.resolve("build")
        .resolve("dist")
        .resolve("js")
        .resolve("productionExecutable")
    println("serve page at ${pagePath.absolutePath}")
    return embeddedServer(Netty, port = 9000) {

        routing {
            staticFiles(
                "/",
                pagePath
            )
        }
    }.start(wait = false)
}


fun browser(projectDir: File, logger: Logger) {
    logger.info("starting browser")

    Playwright.create().use { playwright ->
        val browserTypes: List<BrowserType> = Arrays.asList(
            playwright.chromium(),
            // Not yet suported
            // playwright.webkit(),
            // Not yet suported
            //playwright.firefox()
        )
        for (browserType in browserTypes) {
            logger.info("will run test on ${browserType.name()}")
            browserType.launch().use { browser ->
                logger.info("browser started")
                var renderEnded: Boolean
                val context: BrowserContext = browser.newContext()
                val page: Page = context.newPage()
                page.navigate("chrome://gpu")
                page.screenshot(
                    Page.ScreenshotOptions()
                        .setPath(
                            projectDir
                                .resolve("js-${browserType.name().toString()}")
                                .also { it.mkdirs() }
                                .resolve("gpu.png")
                                .toPath()
                        )
                )
                page.setViewportSize(256, 256)
                context.onConsoleMessage {
                    println(it.text())
                    if (it.text().equals("render ended", ignoreCase = true)) {
                        renderEnded = true
                    }
                }
                try {
                    scenes.forEach { (sceneName, frames) ->
                        frames.forEach { frame ->
                            renderEnded = false
                            page.navigate("http://localhost:9000/index.html?scene=$sceneName&frame=$frame")
                            context.waitForCondition { renderEnded }
                            page.screenshot(
                                Page.ScreenshotOptions()
                                    .setPath(
                                        projectDir
                                            .resolve("js-${browserType.name().toString()}")
                                            .also { it.mkdirs() }
                                            .resolve("$sceneName-$frame.png")
                                            .toPath()
                                    )
                            )
                        }
                    }
                } catch (error: Exception) {
                    logger.info("fail to render on browser", error)
                }


            }
        }
    }

}

