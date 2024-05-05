
plugins {
	`kotlin-dsl`
}

dependencies {
	implementation("de.undercouch:gradle-download-task:5.6.0")
	implementation(libs.kotlinMultiplatformPlugin)
}

repositories {
	mavenCentral()
}