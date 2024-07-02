plugins {
    kotlin("jvm")
}

dependencies {
    implementation(projects.examples.headless)
}

val jvmTask = mutableListOf<TaskProvider<*>>()
scenes.forEach { (sceneName, frames) ->
    frames.forEach { frame ->
        tasks.register<JavaExec>("e2eJvm-$sceneName-$frame") {
            isIgnoreExitValue = true
            mainClass = "MainKt"
            jvmArgs(
                if (Platform.os == Os.MacOs) {
                    listOf(
                        "-XstartOnFirstThread",
                        "--add-opens=java.base/java.lang=ALL-UNNAMED",
                        "--enable-native-access=ALL-UNNAMED"
                    )
                } else {
                    listOf(
                        "--add-opens=java.base/java.lang=ALL-UNNAMED",
                        "--enable-native-access=ALL-UNNAMED"
                    )
                } + listOf(
                    "-Dscene=$sceneName",
                    "-Dframe=$frame",
                    "-DscreenshotPath=${project.projectDir}",
                )
            )
            classpath = sourceSets["main"].runtimeClasspath
        }.also { jvmTask.add(it) }
    }
}

val e2eBrowserTest = tasks.create("e2eBrowserTest") {
    doLast {
        val server = endToEndWebserver(getHeadlessProject().projectDir)
        browser(project.projectDir, logger)
        server.stop()

    }
}


tasks.create("e2eTest") {
    dependsOn(e2eBrowserTest)
}

fun getHeadlessProject() = projects.examples.headless.identityPath.path
    ?.let(::project) ?: error("Could not find project path")