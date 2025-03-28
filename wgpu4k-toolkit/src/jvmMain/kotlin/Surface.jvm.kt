package io.ygdrasil.webgpu

import org.lwjgl.glfw.GLFW.glfwGetWindowSize

actual class Surface(private val handler: NativeSurface, private val windowHandler: Long) : AutoCloseable {
    actual val width: UInt
        get() {
            val width = IntArray(1)
            val height = IntArray(1)
            glfwGetWindowSize(windowHandler, width, height)
            return width[0].toUInt()
        }
    actual val height: UInt
        get() {
            val width = IntArray(1)
            val height = IntArray(1)
            glfwGetWindowSize(windowHandler, width, height)
            return height[0].toUInt()
        }

    actual val supportedFormats: Set<GPUTextureFormat>
        get() = handler.supportedFormats
    actual val supportedAlphaMode: Set<CompositeAlphaMode>
        get() = handler.supportedAlphaMode

    actual fun getCurrentTexture(): SurfaceTexture = handler.getCurrentTexture()

    actual fun present() = handler.present()

    actual fun configure(surfaceConfiguration: SurfaceConfiguration) {
        handler.configure(surfaceConfiguration, width, height)
    }

    actual override fun close() = handler.close()

}