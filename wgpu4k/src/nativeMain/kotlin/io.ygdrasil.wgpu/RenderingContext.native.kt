package io.ygdrasil.wgpu

actual class RenderingContext : AutoCloseable {
    actual val width: Int
        get() = TODO("Not yet implemented")
    actual val height: Int
        get() = TODO("Not yet implemented")
    actual val textureFormat: TextureFormat
        get() = TODO("Not yet implemented")

    actual fun getCurrentTexture(): Texture {
        TODO("Not yet implemented")
    }

    /**
     * Schedule this texture to be presented on the owning surface.
     *
     * Needs to be called after any work on the texture is scheduled via Queue::submit.
     *
     * Platform dependent behavior
     * On Wayland, present will attach a wl_buffer to the underlying wl_surface and commit the new surface state. If it is desired to do things such as request a frame callback, scale the surface using the viewporter or synchronize other double buffered state, then these operations should be done before the call to present.
     */
    actual fun present() {
    }

    actual fun configure(canvasConfiguration: CanvasConfiguration) {
    }

    override fun close() {
        TODO("Not yet implemented")
    }

}