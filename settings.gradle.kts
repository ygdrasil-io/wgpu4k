rootProject.name = "wgpu4k-root"
enableFeaturePreview("TYPESAFE_PROJECT_ACCESSORS")

pluginManagement {
	repositories {
		gradlePluginPortal()
		mavenCentral()
		mavenLocal()
		maven("https://maven.pkg.jetbrains.space/public/p/compose/dev")
	}
}

include("wgpu4k-e2e")
