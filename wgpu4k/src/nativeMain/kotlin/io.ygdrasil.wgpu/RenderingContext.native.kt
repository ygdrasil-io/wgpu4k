@file:OptIn(ExperimentalForeignApi::class)

package io.ygdrasil.wgpu

import kotlinx.cinterop.*
import webgpu.*

actual class RenderingContext(
    internal val handler: WGPUSurface,
    private val sizeProvider: () -> Pair<Int, Int>
) : AutoCloseable {

    private val surfaceCapabilities = cValue<WGPUSurfaceCapabilities>()

    actual val width: Int
        get() = sizeProvider().first
    actual val height: Int
        get() = sizeProvider().second

    actual val textureFormat: TextureFormat by lazy {
        surfaceCapabilities.useContents { formats }?.get(0)?.toInt()
            ?.let { TextureFormat.of(it) ?: error("texture format not found") }
            ?: error("call first computeSurfaceCapabilities")
    }

    actual fun getCurrentTexture(): Texture {
        val surfaceTexture = cValue<WGPUSurfaceTexture>()
        wgpuSurfaceGetCurrentTexture(handler, surfaceTexture)
        return Texture(surfaceTexture.useContents { texture } ?: error("no texture available"))
    }

    actual fun present() {
        wgpuSurfacePresent(handler)
    }

    fun computeSurfaceCapabilities(adapter: Adapter) {
        wgpuSurfaceGetCapabilities(handler, adapter.handler, surfaceCapabilities)
    }

    actual fun configure(canvasConfiguration: CanvasConfiguration) {

        if (surfaceCapabilities.useContents { formats } == null) error("call computeSurfaceCapabilities(adapter: Adapter) before configure")

        wgpuSurfaceConfigure(handler, canvasConfiguration.convert())
    }

    override fun close() {
        wgpuSurfaceRelease(handler)
    }

    private fun CanvasConfiguration.convert(): CValue<WGPUSurfaceConfiguration> = cValue<WGPUSurfaceConfiguration>() {
        device = this@convert.device.handler
        usage = this@convert.usage.toUInt()
        format = (this@convert.format?.value ?: textureFormat.value).toUInt()
        presentMode = WGPUPresentMode_Fifo
        alphaMode = this@convert.alphaMode?.value?.toUInt() ?: surfaceCapabilities.useContents {alphaModes }?.get(0) ?: error("")
        width = this@RenderingContext.width.toUInt()
        height = this@RenderingContext.height.toUInt()
    }

}