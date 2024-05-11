
plugins {
	`kotlin-dsl`
}

dependencies {
	implementation("com.squareup:kotlinpoet:1.16.0")
	implementation("de.undercouch:gradle-download-task:5.6.0")
	implementation(libs.kotlinMultiplatformPlugin)
}

repositories {
	mavenCentral()
}