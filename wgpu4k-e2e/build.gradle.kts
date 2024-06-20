
plugins {
    alias(libs.plugins.kotlinMultiplatform)
	alias(libs.plugins.kotest)
}

java {
	toolchain {
		languageVersion.set(JavaLanguageVersion.of(22))
	}
}

kotlin {

    js { browser() }
    jvm()

    sourceSets {

        val commonMain by getting {
            dependencies {
				implementation(projects.examples.common)
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(libs.bundles.kotest)
            }
        }

        val jvmTest by getting {
            dependencies {
                implementation(libs.kotest.runner.junit5)
            }

        }
    }
    compilerOptions {
        //allWarningsAsErrors = true
    }
}

tasks.named<Test>("jvmTest") {
	useJUnitPlatform()
    jvmArgs = listOf("-XstartOnFirstThread", "--add-opens=java.base/java.lang=ALL-UNNAMED")
	filter {
		isFailOnNoMatchingTests = false
	}
	testLogging {
		showExceptions = true
		showStandardStreams = true
		events = setOf(
			org.gradle.api.tasks.testing.logging.TestLogEvent.FAILED,
			org.gradle.api.tasks.testing.logging.TestLogEvent.PASSED
		)
		exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
	}
}
