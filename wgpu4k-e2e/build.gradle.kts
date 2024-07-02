plugins {
    kotlin("jvm")
}

val e2eBrowserTest = tasks.create("e2eBrowserTest") {
    doLast {
        browser(project.projectDir, logger)
    }
}


tasks.create("e2eTest") {
    dependsOn(e2eBrowserTest)
}
