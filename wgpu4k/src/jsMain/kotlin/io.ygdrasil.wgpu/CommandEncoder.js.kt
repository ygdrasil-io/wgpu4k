

package io.ygdrasil.wgpu

import io.ygdrasil.wgpu.internal.js.*

actual class CommandEncoder(private val handler: GPUCommandEncoder) : AutoCloseable {
	actual fun beginRenderPass(descriptor: RenderPassDescriptor): RenderPassEncoder {
		return RenderPassEncoder(handler.beginRenderPass(descriptor.convert()))
	}

	actual fun finish(): CommandBuffer {
		return CommandBuffer(handler.finish())
	}

	actual fun copyTextureToTexture(
		source: ImageCopyTexture,
		destination: ImageCopyTexture,
		copySize: Size3D
	) {
		handler.copyTextureToTexture(source.convert(), destination.convert(), copySize.toArray())
	}

	actual fun beginComputePass(descriptor: ComputePassDescriptor?): ComputePassEncoder =
		descriptor?.convert()
			.let { handler.beginComputePass(it ?: undefined) }
			.let { ComputePassEncoder(it) }

	actual override fun close() {
		// Nothing to do
	}
}

private fun ComputePassDescriptor?.convert(): GPUComputePassDescriptor {
	TODO()
}

private fun ImageCopyTexture.convert(): GPUImageCopyTexture = object : GPUImageCopyTexture {
	override var texture: GPUTexture = this@convert.texture.handler
	override var mipLevel: GPUIntegerCoordinate = this@convert.mipLevel
	override var origin: dynamic = this@convert.origin.toArray()
	override var aspect: String = this@convert.aspect.stringValue
}


private fun RenderPassDescriptor.convert(): GPURenderPassDescriptor = object : GPURenderPassDescriptor {
	override var colorAttachments: Array<GPURenderPassColorAttachment> =
		this@convert.colorAttachments.map { it.convert() }.toTypedArray()
	override var label: String? = this@convert.label ?: undefined
	override var depthStencilAttachment: GPURenderPassDepthStencilAttachment? =
		this@convert.depthStencilAttachment?.convert() ?: undefined

	/*
	override var occlusionQuerySet: GPUQuerySet?
	override var timestampWrites: GPURenderPassTimestampWrites?
	*/
	override var maxDrawCount: GPUSize64? = this@convert.maxDrawCount
}

private fun RenderPassDescriptor.RenderPassDepthStencilAttachment.convert(): GPURenderPassDepthStencilAttachment =
	object : GPURenderPassDepthStencilAttachment {
		override var view: GPUTextureView = this@convert.view.handler
		override var depthClearValue: Number? = this@convert.depthClearValue ?: undefined
		override var depthLoadOp: String? = this@convert.depthLoadOp?.name ?: undefined
		override var depthStoreOp: String? = this@convert.depthStoreOp?.name ?: undefined
		override var depthReadOnly: Boolean? = this@convert.depthReadOnly
		override var stencilClearValue: GPUStencilValue? = this@convert.stencilClearValue
		override var stencilLoadOp: String? = this@convert.stencilLoadOp?.name ?: undefined
		override var stencilStoreOp: String? = this@convert.stencilStoreOp?.name ?: undefined
		override var stencilReadOnly: Boolean? = this@convert.stencilReadOnly
	}

private fun RenderPassDescriptor.ColorAttachment.convert(): GPURenderPassColorAttachment =
	object : GPURenderPassColorAttachment {
		override var view: GPUTextureView = this@convert.view.handler
		override var loadOp: String = this@convert.loadOp.name
		override var storeOp: String = this@convert.storeOp.name
		override var depthSlice: GPUIntegerCoordinate? = this@convert.depthSlice ?: undefined
		override var resolveTarget: GPUTextureView? = this@convert.resolveTarget?.handler ?: undefined
		override var clearValue: Array<Number>? = this@convert.clearValue
	}
