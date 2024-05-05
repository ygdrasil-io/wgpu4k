@file:OptIn(ExperimentalForeignApi::class)

package io.ygdrasil.wgpu

import kotlinx.cinterop.*
import webgpu.*

class WGPU(val handler: WGPUInstance) {

    companion object {

        fun createInstance(backend: WGPUInstanceBackend? = null): WGPU? = memScoped {
            wgpuCreateInstance(getDescriptor(backend))
                ?.let { WGPU(it) }
        }

        private fun MemScope.getDescriptor(backend: WGPUInstanceBackend?): CValue<WGPUInstanceDescriptor>? {
            if (backend == null) return null

            val temp = cValue<WGPUInstanceExtras> {
                chain.sType = WGPUSType_InstanceExtras
                backends = backend
            }
            val descriptor = cValue<WGPUInstanceDescriptor> {
                nextInChain = temp.ptr.reinterpret()
            }


            return descriptor
        }
    }
}