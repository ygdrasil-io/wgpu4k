@file:OptIn(ExperimentalForeignApi::class, BetaInteropApi::class)

import io.ygdrasil.wgpu.RenderingContext
import io.ygdrasil.wgpu.WGPU.Companion.createInstance
import kotlinx.cinterop.*
import kotlinx.coroutines.runBlocking
import platform.AppKit.*
import platform.CoreGraphics.CGSize
import platform.Foundation.NSMakeRect
import platform.Foundation.NSNotification
import platform.Foundation.NSRect
import platform.Metal.MTLClearColorMake
import platform.Metal.MTLCreateSystemDefaultDevice
import platform.Metal.MTLPixelFormatBGRA8Unorm_sRGB
import platform.MetalKit.MTKView
import platform.MetalKit.MTKViewDelegateProtocol
import platform.QuartzCore.CAMetalLayer
import platform.darwin.NSObject
import platform.foundation.height
import platform.foundation.width

val windowStyle = NSWindowStyleMaskTitled or NSWindowStyleMaskMiniaturizable or
        NSWindowStyleMaskClosable or NSWindowStyleMaskResizable or NSBackingStoreBuffered

fun main() {
    Application("hello")
        .run()


}

class Application(
    private val windowTitle: String
) {

    fun run() {
        autoreleasepool {
            val application = NSApplication.sharedApplication()

            val windowRect: CValue<NSRect> = run {
                val frame = NSScreen.mainScreen()!!.frame
                NSMakeRect(
                    .0, .0,
                    frame.width * 0.5,
                    frame.height * 0.5
                )
            }
            val window = NSWindow(windowRect, windowStyle, NSBackingStoreBuffered, false)
            window.contentView()?.setWantsLayer(true)
            val layer = CAMetalLayer.layer()
            window.contentView()?.setLayer(layer)

            application.delegate = object : NSObject(), NSApplicationDelegateProtocol {

                override fun applicationShouldTerminateAfterLastWindowClosed(sender: NSApplication): Boolean {
                    println("applicationShouldTerminateAfterLastWindowClosed")
                    return true
                }

                override fun applicationWillFinishLaunching(notification: NSNotification) {
                    println("applicationWillFinishLaunching")
                }

                override fun applicationDidFinishLaunching(notification: NSNotification) {
                    println("applicationDidFinishLaunching")

                    window.setTitle(windowTitle)

                    window.orderFrontRegardless()
                    window.center()


                    val wgpu = createInstance() ?: error("fail to wgpu instance")
                    val surface =
                        wgpu.getSurfaceFromMetalLayer(interpretCPointer<COpaque>(layer.objcPtr())!!.reinterpret())
                            ?: error("fail to get wgpu surface")


                    val renderingContext = RenderingContext(surface) {
                        window.frame.width.toInt() to window.frame.height.toInt()
                    }

                    val adapter = wgpu.requestAdapter(renderingContext)
                        ?: error("fail to get adapter")

                    val device =  runBlocking { adapter.requestDevice() }
                        ?: error("fail to get device")

                    renderingContext.computeSurfaceCapabilities(adapter)
                }

                override fun applicationWillTerminate(notification: NSNotification) {
                    println("applicationWillTerminate")
                }
            }

            application.run()

        }
    }
}
