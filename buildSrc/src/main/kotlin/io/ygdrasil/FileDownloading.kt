package io.ygdrasil

import de.undercouch.gradle.tasks.download.Download
import org.gradle.api.Project
import org.gradle.api.Task
import org.gradle.api.tasks.Copy
import org.gradle.kotlin.dsl.register
import java.io.File


data class NativeLibrary(val remoteFile: String, val extractedFiles: List<Pair<File, String>>)

fun nativeLibrary(remoteFile: String, extractedFiles: List<Pair<File, String>>)
        = NativeLibrary(remoteFile, extractedFiles)

fun nativeLibrary(remoteFile: String, targetFile: File,  zipFileName: String)
        = NativeLibrary(remoteFile, listOf(targetFile to zipFileName))

fun Project.unzipTask(
    zipFile: File,
    target: File,
    zipFilename: String,
    downloadTask: Task
) = tasks.register<Copy>("unzip-${zipFilename.replace("/", "-")}-from-${zipFile.name}") {
    onlyIf { !target.exists() }
    from(zipTree(zipFile))
    include(zipFilename)
    into(target.parent)
    rename { fileName ->
        fileName.replace(zipFilename, target.name)
    }
    dependsOn(downloadTask)
}.get()


fun Project.downloadInto(baseUrl: String, fileName: String, target: File): Task {
    val url = "$baseUrl$fileName"
    val taskName = "downloadFile-$fileName"
    target.parentFile.mkdirs()
    return tasks.register<Download>(taskName) {
        onlyIf { !target.exists() }
        src(url)
        dest(target)
    }.get()
}