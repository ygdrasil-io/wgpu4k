// Generated by jextract
package io.ygdrasil.wgpu.internal.jna

import java.lang.foreign.AddressLayout
import java.lang.foreign.Arena
import java.lang.foreign.GroupLayout
import java.lang.foreign.MemoryLayout.Companion.sequenceLayout
import java.lang.foreign.MemoryLayout.Companion.structLayout
import java.lang.foreign.MemoryLayout.PathElement.groupElement
import java.lang.foreign.MemorySegment
import java.lang.foreign.SegmentAllocator
import java.lang.foreign.ValueLayout
import java.util.function.Consumer

/**
 * {@snippet lang=c :
 * * struct WGPUSurfaceConfiguration {
 * *     const WGPUChainedStruct *nextInChain;
 * *     WGPUDevice device;
 * *     WGPUTextureFormat format;
 * *     WGPUTextureUsageFlags usage;
 * *     size_t viewFormatCount;
 * *     const WGPUTextureFormat *viewFormats;
 * *     WGPUCompositeAlphaMode alphaMode;
 * *     uint32_t width;
 * *     uint32_t height;
 * *     WGPUPresentMode presentMode;
 * * }
 * * }
 */
object WGPUSurfaceConfiguration {
    private val `$LAYOUT` = structLayout(
        wgpu_h.C_POINTER.withName("nextInChain"),
        wgpu_h.C_POINTER.withName("device"),
        wgpu_h.C_INT.withName("format"),
        wgpu_h.C_INT.withName("usage"),
        wgpu_h.C_LONG.withName("viewFormatCount"),
        wgpu_h.C_POINTER.withName("viewFormats"),
        wgpu_h.C_INT.withName("alphaMode"),
        wgpu_h.C_INT.withName("width"),
        wgpu_h.C_INT.withName("height"),
        wgpu_h.C_INT.withName("presentMode")
    ).withName("WGPUSurfaceConfiguration")

    /**
     * The layout of this struct
     */
    fun layout(): GroupLayout {
        return `$LAYOUT`
    }

    private val `nextInChain$LAYOUT`: AddressLayout = `$LAYOUT`.select(groupElement("nextInChain")) as AddressLayout

    /**
     * Layout for field:
     * {@snippet lang=c :
     * * const WGPUChainedStruct *nextInChain
     * * }
     */
    fun `nextInChain$layout`(): AddressLayout {
        return `nextInChain$LAYOUT`
    }

    private const val `nextInChain$OFFSET`: Long = 0

    /**
     * Offset for field:
     * {@snippet lang=c :
     * * const WGPUChainedStruct *nextInChain
     * * }
     */
    fun `nextInChain$offset`(): Long {
        return `nextInChain$OFFSET`
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * * const WGPUChainedStruct *nextInChain
     * * }
     */
    fun nextInChain(struct: MemorySegment): MemorySegment {
        return struct.get(`nextInChain$LAYOUT`, `nextInChain$OFFSET`)
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * * const WGPUChainedStruct *nextInChain
     * * }
     */
    fun nextInChain(struct: MemorySegment, fieldValue: MemorySegment?) {
        struct.set(
            `nextInChain$LAYOUT`, `nextInChain$OFFSET`,
            fieldValue!!
        )
    }

    private val `device$LAYOUT`: AddressLayout = `$LAYOUT`.select(groupElement("device")) as AddressLayout

    /**
     * Layout for field:
     * {@snippet lang=c :
     * * WGPUDevice device
     * * }
     */
    fun `device$layout`(): AddressLayout {
        return `device$LAYOUT`
    }

    private const val `device$OFFSET`: Long = 8

    /**
     * Offset for field:
     * {@snippet lang=c :
     * * WGPUDevice device
     * * }
     */
    fun `device$offset`(): Long {
        return `device$OFFSET`
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * * WGPUDevice device
     * * }
     */
    fun device(struct: MemorySegment): MemorySegment {
        return struct.get(`device$LAYOUT`, `device$OFFSET`)
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * * WGPUDevice device
     * * }
     */
    fun device(struct: MemorySegment, fieldValue: MemorySegment?) {
        struct.set(
            `device$LAYOUT`, `device$OFFSET`,
            fieldValue!!
        )
    }

    private val `format$LAYOUT` = `$LAYOUT`.select(groupElement("format")) as ValueLayout.OfInt

    /**
     * Layout for field:
     * {@snippet lang=c :
     * * WGPUTextureFormat format
     * * }
     */
    fun `format$layout`(): ValueLayout.OfInt {
        return `format$LAYOUT`
    }

    private const val `format$OFFSET`: Long = 16

    /**
     * Offset for field:
     * {@snippet lang=c :
     * * WGPUTextureFormat format
     * * }
     */
    fun `format$offset`(): Long {
        return `format$OFFSET`
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * * WGPUTextureFormat format
     * * }
     */
    fun format(struct: MemorySegment): Int {
        return struct.get(`format$LAYOUT`, `format$OFFSET`)
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * * WGPUTextureFormat format
     * * }
     */
    fun format(struct: MemorySegment, fieldValue: Int) {
        struct.set(`format$LAYOUT`, `format$OFFSET`, fieldValue)
    }

    private val `usage$LAYOUT` = `$LAYOUT`.select(groupElement("usage")) as ValueLayout.OfInt

    /**
     * Layout for field:
     * {@snippet lang=c :
     * * WGPUTextureUsageFlags usage
     * * }
     */
    fun `usage$layout`(): ValueLayout.OfInt {
        return `usage$LAYOUT`
    }

    private const val `usage$OFFSET`: Long = 20

    /**
     * Offset for field:
     * {@snippet lang=c :
     * * WGPUTextureUsageFlags usage
     * * }
     */
    fun `usage$offset`(): Long {
        return `usage$OFFSET`
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * * WGPUTextureUsageFlags usage
     * * }
     */
    fun usage(struct: MemorySegment): Int {
        return struct.get(`usage$LAYOUT`, `usage$OFFSET`)
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * * WGPUTextureUsageFlags usage
     * * }
     */
    fun usage(struct: MemorySegment, fieldValue: Int) {
        struct.set(`usage$LAYOUT`, `usage$OFFSET`, fieldValue)
    }

    private val `viewFormatCount$LAYOUT` = `$LAYOUT`.select(groupElement("viewFormatCount")) as ValueLayout.OfLong

    /**
     * Layout for field:
     * {@snippet lang=c :
     * * size_t viewFormatCount
     * * }
     */
    fun `viewFormatCount$layout`(): ValueLayout.OfLong {
        return `viewFormatCount$LAYOUT`
    }

    private const val `viewFormatCount$OFFSET`: Long = 24

    /**
     * Offset for field:
     * {@snippet lang=c :
     * * size_t viewFormatCount
     * * }
     */
    fun `viewFormatCount$offset`(): Long {
        return `viewFormatCount$OFFSET`
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * * size_t viewFormatCount
     * * }
     */
    fun viewFormatCount(struct: MemorySegment): Long {
        return struct.get(`viewFormatCount$LAYOUT`, `viewFormatCount$OFFSET`)
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * * size_t viewFormatCount
     * * }
     */
    fun viewFormatCount(struct: MemorySegment, fieldValue: Long) {
        struct.set(`viewFormatCount$LAYOUT`, `viewFormatCount$OFFSET`, fieldValue)
    }

    private val `viewFormats$LAYOUT`: AddressLayout = `$LAYOUT`.select(groupElement("viewFormats")) as AddressLayout

    /**
     * Layout for field:
     * {@snippet lang=c :
     * * const WGPUTextureFormat *viewFormats
     * * }
     */
    fun `viewFormats$layout`(): AddressLayout {
        return `viewFormats$LAYOUT`
    }

    private const val `viewFormats$OFFSET`: Long = 32

    /**
     * Offset for field:
     * {@snippet lang=c :
     * * const WGPUTextureFormat *viewFormats
     * * }
     */
    fun `viewFormats$offset`(): Long {
        return `viewFormats$OFFSET`
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * * const WGPUTextureFormat *viewFormats
     * * }
     */
    fun viewFormats(struct: MemorySegment): MemorySegment {
        return struct.get(`viewFormats$LAYOUT`, `viewFormats$OFFSET`)
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * * const WGPUTextureFormat *viewFormats
     * * }
     */
    fun viewFormats(struct: MemorySegment, fieldValue: MemorySegment?) {
        struct.set(
            `viewFormats$LAYOUT`, `viewFormats$OFFSET`,
            fieldValue!!
        )
    }

    private val `alphaMode$LAYOUT` = `$LAYOUT`.select(groupElement("alphaMode")) as ValueLayout.OfInt

    /**
     * Layout for field:
     * {@snippet lang=c :
     * * WGPUCompositeAlphaMode alphaMode
     * * }
     */
    fun `alphaMode$layout`(): ValueLayout.OfInt {
        return `alphaMode$LAYOUT`
    }

    private const val `alphaMode$OFFSET`: Long = 40

    /**
     * Offset for field:
     * {@snippet lang=c :
     * * WGPUCompositeAlphaMode alphaMode
     * * }
     */
    fun `alphaMode$offset`(): Long {
        return `alphaMode$OFFSET`
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * * WGPUCompositeAlphaMode alphaMode
     * * }
     */
    fun alphaMode(struct: MemorySegment): Int {
        return struct.get(`alphaMode$LAYOUT`, `alphaMode$OFFSET`)
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * * WGPUCompositeAlphaMode alphaMode
     * * }
     */
    fun alphaMode(struct: MemorySegment, fieldValue: Int) {
        struct.set(`alphaMode$LAYOUT`, `alphaMode$OFFSET`, fieldValue)
    }

    private val `width$LAYOUT` = `$LAYOUT`.select(groupElement("width")) as ValueLayout.OfInt

    /**
     * Layout for field:
     * {@snippet lang=c :
     * * uint32_t width
     * * }
     */
    fun `width$layout`(): ValueLayout.OfInt {
        return `width$LAYOUT`
    }

    private const val `width$OFFSET`: Long = 44

    /**
     * Offset for field:
     * {@snippet lang=c :
     * * uint32_t width
     * * }
     */
    fun `width$offset`(): Long {
        return `width$OFFSET`
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * * uint32_t width
     * * }
     */
    fun width(struct: MemorySegment): Int {
        return struct.get(`width$LAYOUT`, `width$OFFSET`)
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * * uint32_t width
     * * }
     */
    fun width(struct: MemorySegment, fieldValue: Int) {
        struct.set(`width$LAYOUT`, `width$OFFSET`, fieldValue)
    }

    private val `height$LAYOUT` = `$LAYOUT`.select(groupElement("height")) as ValueLayout.OfInt

    /**
     * Layout for field:
     * {@snippet lang=c :
     * * uint32_t height
     * * }
     */
    fun `height$layout`(): ValueLayout.OfInt {
        return `height$LAYOUT`
    }

    private const val `height$OFFSET`: Long = 48

    /**
     * Offset for field:
     * {@snippet lang=c :
     * * uint32_t height
     * * }
     */
    fun `height$offset`(): Long {
        return `height$OFFSET`
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * * uint32_t height
     * * }
     */
    fun height(struct: MemorySegment): Int {
        return struct.get(`height$LAYOUT`, `height$OFFSET`)
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * * uint32_t height
     * * }
     */
    fun height(struct: MemorySegment, fieldValue: Int) {
        struct.set(`height$LAYOUT`, `height$OFFSET`, fieldValue)
    }

    private val `presentMode$LAYOUT` = `$LAYOUT`.select(groupElement("presentMode")) as ValueLayout.OfInt

    /**
     * Layout for field:
     * {@snippet lang=c :
     * * WGPUPresentMode presentMode
     * * }
     */
    fun `presentMode$layout`(): ValueLayout.OfInt {
        return `presentMode$LAYOUT`
    }

    private const val `presentMode$OFFSET`: Long = 52

    /**
     * Offset for field:
     * {@snippet lang=c :
     * * WGPUPresentMode presentMode
     * * }
     */
    fun `presentMode$offset`(): Long {
        return `presentMode$OFFSET`
    }

    /**
     * Getter for field:
     * {@snippet lang=c :
     * * WGPUPresentMode presentMode
     * * }
     */
    fun presentMode(struct: MemorySegment): Int {
        return struct.get(`presentMode$LAYOUT`, `presentMode$OFFSET`)
    }

    /**
     * Setter for field:
     * {@snippet lang=c :
     * * WGPUPresentMode presentMode
     * * }
     */
    fun presentMode(struct: MemorySegment, fieldValue: Int) {
        struct.set(`presentMode$LAYOUT`, `presentMode$OFFSET`, fieldValue)
    }

    /**
     * Obtains a slice of `arrayParam` which selects the array element at `index`.
     * The returned segment has address `arrayParam.address() + index * layout().byteSize()`
     */
    fun asSlice(array: MemorySegment, index: Long): MemorySegment {
        return array.asSlice(layout().byteSize() * index)
    }

    /**
     * The size (in bytes) of this struct
     */
    fun sizeof(): Long {
        return layout().byteSize()
    }

    /**
     * Allocate a segment of size `layout().byteSize()` using `allocator`
     */
    fun allocate(allocator: SegmentAllocator): MemorySegment {
        return allocator.allocate(layout())
    }

    /**
     * Allocate an array of size `elementCount` using `allocator`.
     * The returned segment has size `elementCount * layout().byteSize()`.
     */
    fun allocateArray(elementCount: Long, allocator: SegmentAllocator): MemorySegment {
        return allocator.allocate(sequenceLayout(elementCount, layout()))
    }

    /**
     * Reinterprets `addr` using target `arena` and `cleanupAction` (if any).
     * The returned segment has size `layout().byteSize()`
     */
    fun reinterpret(addr: MemorySegment, arena: Arena, cleanup: Consumer<MemorySegment?>): MemorySegment {
        return reinterpret(addr, 1, arena, cleanup)
    }

    /**
     * Reinterprets `addr` using target `arena` and `cleanupAction` (if any).
     * The returned segment has size `elementCount * layout().byteSize()`
     */
    fun reinterpret(
        addr: MemorySegment,
        elementCount: Long,
        arena: Arena,
        cleanup: Consumer<MemorySegment?>
    ): MemorySegment {
        return addr.reinterpret(layout().byteSize() * elementCount, arena, cleanup)
    }
}
