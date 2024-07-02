plugins {
	`kotlin-dsl`
}


repositories {
	mavenCentral()
	maven( url = "https://jitpack.io" )
}

dependencies {
	implementation("com.microsoft.playwright:playwright:1.41.0")
}
