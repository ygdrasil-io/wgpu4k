import de.undercouch.gradle.tasks.download.Download
import io.github.krakowski.jextract.JextractTask
import io.ygdrasil.wathever
import org.jetbrains.kotlin.gradle.tasks.KotlinCompileCommon

plugins {
    alias(libs.plugins.kotlinMultiplatform)
	alias(libs.plugins.kotest)
	id("io.github.krakowski.jextract") version "0.5.0" apply false
	alias(libs.plugins.download)
}

java {
	toolchain {
		languageVersion.set(JavaLanguageVersion.of(22))
	}
}

// You need to use a JDK version with jextract from here
// https://jdk.java.net/jextract/
val jextract = tasks.withType<JextractTask> {
	header("${project.projectDir}/../headers/wgpu.h") {

		// The package under which all source files will be generated
		targetPackage = "io.ygdrasil.wgpu.internal.jvm.panama"

		outputDir = project.objects.directoryProperty()
			.convention(project.layout.projectDirectory.dir("src/jvmMain"))
	}
}



val buildNativeResourcesDirectory = project.file("build").resolve("native")

kotlin {

    js {
        binaries.executable()
        browser()
        nodejs()
    }
    jvm {
        withJava()
    }

    val target = setOf(
        macosArm64(),
        macosX64(),
        /*mingwX64(),
        linuxX64(),
        linuxArm64()*/
    )

    target.forEach {
        with(it) {
            val main by compilations.getting {
                cinterops.create("webgpu") {
                    header(buildNativeResourcesDirectory.resolve("wgpu.h"))
                }
            }

        }
    }

    sourceSets {

        all {
            languageSettings.optIn("kotlin.ExperimentalStdlibApi")
            languageSettings.optIn("kotlin.ExperimentalUnsignedTypes")
            languageSettings.optIn("kotlin.js.ExperimentalJsExport")
        }

        val kotlinWrappersVersion = "1.0.0-pre.721"

        val jsMain by getting {
            dependencies {
                implementation(project.dependencies.platform("org.jetbrains.kotlin-wrappers:kotlin-wrappers-bom:$kotlinWrappersVersion"))
                implementation("org.jetbrains.kotlin-wrappers:kotlin-js")
                implementation("org.jetbrains.kotlin-wrappers:kotlin-web")
            }
        }

        val jvmMain by getting {
            dependencies {
                kotlin.srcDirs("src/jvmMain/kotlin", "src/jvmMain/java")
            }
        }

        val commonMain by getting {
            dependencies {
                implementation(kotlin("stdlib-common"))
                implementation(libs.coroutines)
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
        allWarningsAsErrors = true
        freeCompilerArgs.add("-Xexpect-actual-classes")
    }
}

tasks {
	withType<JavaCompile> {
		options.compilerArgs.add("--enable-preview")
	}
}

val resourcesDirectory = project.file("src").resolve("jvmMain").resolve("resources")

wathever {
    baseUrl = "https://github.com/gfx-rs/wgpu-native/releases/download/${libs.versions.wgpu.get()}/"

    download("wgpu-macos-aarch64-release.zip") {
        extract("libwgpu_native.dylib", resourcesDirectory.resolve("darwin-aarch64").resolve("libWGPU.dylib"))
        extract("webgpu.h", buildNativeResourcesDirectory.resolve("webgpu.h"))
        extract("wgpu.h", buildNativeResourcesDirectory.resolve("wgpu.h"))
        extract("libwgpu_native.a", buildNativeResourcesDirectory.resolve("darwin-aarch64").resolve("libWGPU.a"))
    }

    download("wgpu-macos-x86_64-release.zip") {
        extract("libwgpu_native.dylib", resourcesDirectory.resolve("darwin-x86-64").resolve("libWGPU.dylib"))
        extract("libwgpu_native.a", buildNativeResourcesDirectory.resolve("darwin-x64").resolve("libWGPU.a"))
    }

    download("wgpu-windows-x86_64-release.zip") {
        extract("wgpu_native.dll", resourcesDirectory.resolve("win32-x86-64").resolve("WGPU.dll"))
    }

    download("wgpu-linux-x86_64-release.zip") {
        extract("libwgpu_native.so", resourcesDirectory.resolve("linux-x86-64").resolve("libWGPU.so"))
    }

    download("wgpu-linux-aarch64-release.zip") {
        extract("libwgpu_native.so", resourcesDirectory.resolve("linux-aarch64").resolve("libWGPU.so"))
    }
}

tasks.named<Test>("jvmTest") {
	useJUnitPlatform()
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