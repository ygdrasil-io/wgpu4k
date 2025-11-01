package io.ygdrasil.webgpu.examples.scenes.basic

import io.ygdrasil.webgpu.AutoClosableContext
import io.ygdrasil.webgpu.BindGroupDescriptor
import io.ygdrasil.webgpu.BindGroupEntry
import io.ygdrasil.webgpu.BufferDescriptor
import io.ygdrasil.webgpu.Color
import io.ygdrasil.webgpu.ColorTargetState
import io.ygdrasil.webgpu.Extent3D
import io.ygdrasil.webgpu.FragmentState
import io.ygdrasil.webgpu.GPUBindGroup
import io.ygdrasil.webgpu.GPUBuffer
import io.ygdrasil.webgpu.GPUBufferUsage
import io.ygdrasil.webgpu.GPUFilterMode
import io.ygdrasil.webgpu.GPUIndexFormat
import io.ygdrasil.webgpu.GPULoadOp
import io.ygdrasil.webgpu.GPUPrimitiveTopology
import io.ygdrasil.webgpu.GPURenderPipeline
import io.ygdrasil.webgpu.GPUStoreOp
import io.ygdrasil.webgpu.GPUTexture
import io.ygdrasil.webgpu.GPUTextureFormat
import io.ygdrasil.webgpu.GPUTextureUsage
import io.ygdrasil.webgpu.GPUVertexFormat
import io.ygdrasil.webgpu.PrimitiveState
import io.ygdrasil.webgpu.RenderPassColorAttachment
import io.ygdrasil.webgpu.RenderPassDescriptor
import io.ygdrasil.webgpu.RenderPipelineDescriptor
import io.ygdrasil.webgpu.SamplerDescriptor
import io.ygdrasil.webgpu.ShaderModuleDescriptor
import io.ygdrasil.webgpu.TexelCopyBufferLayout
import io.ygdrasil.webgpu.TexelCopyTextureInfo
import io.ygdrasil.webgpu.TextureDescriptor
import io.ygdrasil.webgpu.VertexAttribute
import io.ygdrasil.webgpu.VertexBufferLayout
import io.ygdrasil.webgpu.VertexState
import io.ygdrasil.webgpu.WGPUContext
import io.ygdrasil.webgpu.beginRenderPass
import io.ygdrasil.webgpu.examples.Scene
import io.ygdrasil.webgpu.writeInto
import io.ygdrasil.webgpu.writeTexture

/**
 * A test to isolate texture writeTexture issues by using hardcoded data instead of VFS loading.
 */
class HardCodedTextureScene(wgpuContext: WGPUContext) : Scene(wgpuContext) {

    // language=wgsl
    private val textureShader = """
         struct VertexOutput {
             @location(0) uv: vec2<f32>,
             @builtin(position) position: vec4<f32>,
         };
         
         @vertex
         fn vs_main(
             @location(0) pos: vec2<f32>,
             @location(1) uvs: vec2<f32>) -> VertexOutput {
             var output: VertexOutput;
             output.position = vec4<f32>(pos.x, pos.y, 0.0, 1.0);
             output.uv = uvs;
             return output;
         }
         
         @group(0) @binding(0)
         var my_texture: texture_2d<f32>;
         
         @group(0) @binding(1)
         var my_sampler: sampler;
         
         @fragment
         fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
             return textureSample(my_texture, my_sampler, in.uv);
         }
     """.trimIndent()

    lateinit var renderPipeline: GPURenderPipeline
    lateinit var vertexBuffer: GPUBuffer
    lateinit var indexBuffer: GPUBuffer
    lateinit var texture: GPUTexture
    lateinit var bindGroup: GPUBindGroup

    override suspend fun initialize() = with(autoClosableContext) {

        // Vertex data: position (x, y) + UV (u, v)
        val vertices = floatArrayOf(
            -0.5f, -0.5f, 0f, 1f,  // bottom-left
            -0.5f,  0.5f, 0f, 0f,  // top-left
            0.5f,  0.5f, 1f, 0f,  // top-right
            0.5f, -0.5f, 1f, 1f,  // bottom-right
        )

        val indices = shortArrayOf(0, 1, 2, 0, 2, 3)

        // Create vertex buffer
        vertexBuffer = device.createBuffer(
            BufferDescriptor(
                size = (vertices.size * Float.SIZE_BYTES).toULong(),
                usage = setOf(GPUBufferUsage.Vertex, GPUBufferUsage.CopyDst),
                mappedAtCreation = true
            )
        ).bind()
        vertices.writeInto(vertexBuffer.getMappedRange())
        vertexBuffer.unmap()

        // Create index buffer
        indexBuffer = device.createBuffer(
            BufferDescriptor(
                size = (indices.size * Short.SIZE_BYTES).toULong(),
                usage = setOf(GPUBufferUsage.Index, GPUBufferUsage.CopyDst),
                mappedAtCreation = true
            )
        ).bind()
        indices.writeInto(indexBuffer.getMappedRange())
        indexBuffer.unmap()

        // Create a simple 2x2 RGBA texture with hardcoded data
        val textureWidth = 2
        val textureHeight = 2
        val textureData = byteArrayOf(
            // Pixel 0,0: Red
            255.toByte(), 0, 0, 255.toByte(),
            // Pixel 1,0: Green
            0, 255.toByte(), 0, 255.toByte(),
            // Pixel 0,1: Blue
            0, 0, 255.toByte(), 255.toByte(),
            // Pixel 1,1: White
            255.toByte(), 255.toByte(), 255.toByte(), 255.toByte()
        )

        // Determine texture format based on preference
        val textureFormat = if (renderingContext.textureFormat.toString().contains("srgb", ignoreCase = true)) {
            GPUTextureFormat.RGBA8UnormSrgb
        } else {
            GPUTextureFormat.RGBA8Unorm
        }

        texture = device.createTexture(
            TextureDescriptor(
                size = Extent3D(textureWidth.toUInt(), textureHeight.toUInt(), 1u),
                mipLevelCount = 1u,
                sampleCount = 1u,
                format = textureFormat,
                usage = setOf(GPUTextureUsage.CopyDst, GPUTextureUsage.TextureBinding),
            )
        ).bind()

        // Write texture data
        @Suppress("DEPRECATION")
        device.queue.writeTexture(
            destination = TexelCopyTextureInfo(texture = texture),
            data = textureData,
            dataLayout = TexelCopyBufferLayout(
                offset = 0u,
                bytesPerRow = (textureWidth * 4).toUInt(),
                rowsPerImage = textureHeight.toUInt()
            ),
            size = Extent3D(textureWidth.toUInt(), textureHeight.toUInt(), 1u)
        )

        // Create sampler
        val sampler = device.createSampler(
            SamplerDescriptor(
                magFilter = GPUFilterMode.Nearest,
                minFilter = GPUFilterMode.Nearest,
            )
        ).bind()

        // Create shader module
        val shaderModule = device.createShaderModule(
            ShaderModuleDescriptor(code = textureShader)
        ).bind()

        // Create render pipeline
        renderPipeline = device.createRenderPipeline(
            RenderPipelineDescriptor(
                vertex = VertexState(
                    module = shaderModule,
                    entryPoint = "vs_main",
                    buffers = listOf(
                        VertexBufferLayout(
                            arrayStride = (4 * Float.SIZE_BYTES).toULong(),
                            attributes = listOf(
                                VertexAttribute(
                                    format = GPUVertexFormat.Float32x2,
                                    offset = 0u,
                                    shaderLocation = 0u,
                                ),
                                VertexAttribute(
                                    format = GPUVertexFormat.Float32x2,
                                    offset = (2 * Float.SIZE_BYTES).toULong(),
                                    shaderLocation = 1u,
                                ),
                            ),
                        )
                    ),
                ),
                fragment = FragmentState(
                    module = shaderModule,
                    entryPoint = "fs_main",
                    targets = listOf(
                        ColorTargetState(
                            format = renderingContext.textureFormat
                        )
                    ),
                ),
                primitive = PrimitiveState(
                    topology = GPUPrimitiveTopology.TriangleList,
                ),
            )
        ).bind()

        // Create bind group
        val textureView = texture.createView().bind()
        bindGroup = device.createBindGroup(
            BindGroupDescriptor(
                layout = renderPipeline.getBindGroupLayout(0u),
                entries = listOf(
                    BindGroupEntry(
                        binding = 0u,
                        resource = textureView
                    ),
                    BindGroupEntry(
                        binding = 1u,
                        resource = sampler
                    ),
                )
            )
        ).bind()
    }

    override suspend fun AutoClosableContext.render() {
        val encoder = device.createCommandEncoder().bind()
        val currentTexture = renderingContext.getCurrentTexture().bind()

        encoder.beginRenderPass(
            RenderPassDescriptor(
                colorAttachments = listOf(
                    RenderPassColorAttachment(
                        view = currentTexture.createView().bind(),
                        loadOp = GPULoadOp.Clear,
                        clearValue = Color(0.0, 0.0, 0.0, 1.0),
                        storeOp = GPUStoreOp.Store
                    )
                )
            )
        ) {
            setPipeline(renderPipeline)
            setBindGroup(0u, bindGroup)
            setVertexBuffer(0u, vertexBuffer)
            setIndexBuffer(indexBuffer, GPUIndexFormat.Uint16)
            drawIndexed(6u)
            end()
        }

        val commandBuffer = encoder.finish().bind()
        device.queue.submit(listOf(commandBuffer))
    }
}