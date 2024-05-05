@file:OptIn(ExperimentalForeignApi::class, BetaInteropApi::class)

import io.ygdrasil.wgpu.WGPU.Companion.createInstance
import kotlinx.cinterop.BetaInteropApi
import kotlinx.cinterop.CValue
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.autoreleasepool
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
            val device = MTLCreateSystemDefaultDevice() ?: error("fail to create device")
            //val renderer = rendererProvider(device)
            val mtkView = MTKView(window.frame, device).apply {
                colorPixelFormat = MTLPixelFormatBGRA8Unorm_sRGB
                clearColor = MTLClearColorMake(1.0, 0.0, 0.0, 1.0)
            }

            val wgpu = createInstance() ?: error("fail to wgpu instance")
            println(wgpu)

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

                    mtkView.delegate = object : NSObject(), MTKViewDelegateProtocol {
                        override fun drawInMTKView(view: MTKView) {
                            //renderer.drawOnView(view)
                        }

                        override fun mtkView(view: MTKView, drawableSizeWillChange: CValue<CGSize>) {

                        }

                    }

                    window.setContentView(mtkView)
                    window.setTitle(windowTitle)

                    window.orderFrontRegardless()
                    window.center()
                }

                override fun applicationWillTerminate(notification: NSNotification) {
                    println("applicationWillTerminate")
                }
            }

            application.run()
        }
    }
}