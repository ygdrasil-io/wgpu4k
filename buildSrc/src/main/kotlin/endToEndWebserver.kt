import com.microsoft.playwright.*
import org.gradle.api.logging.Logger
import java.util.*


fun browser(logger: Logger) {
    logger.info("starting browser")

    Playwright.create().use { playwright ->
        logger.info("Playwright created")
        val browserTypes: List<BrowserType> = Arrays.asList(
            playwright.chromium(),
        )
        for (browserType in browserTypes) {
            logger.info("will run test on ${browserType.name()}")
            browserType.launch().use { browser ->
                logger.info("browser started")
                val context: BrowserContext = browser.newContext()
                val page: Page = context.newPage()
                page.navigate("chrome://gpu")
            }
        }
    }

}

