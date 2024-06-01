import org.khronos.webgl.ArrayBuffer
import org.khronos.webgl.Float32Array

val DEFAULT_ORIENTATION = Float32Array(arrayOf(0f, 0f, 0f, 0f))

@OptIn(ExperimentalJsExport::class)
@JsExport
val DEFAULT_POSITION = MyVector3(0.0, 0.0, 0.0)

@OptIn(ExperimentalJsExport::class)
@JsExport
val DEFAULT_SCALE = MyVector3(1.0, 1.0, 1.0)

@OptIn(ExperimentalJsExport::class)
@JsExport
interface TransformOptions {
    val externalStorage: dynamic
    val position: Array<Float>?
    val orientation: Float32Array?
}

@OptIn(ExperimentalJsExport::class)
@JsExport
class TransformKt(
    options: TransformOptions?
) {

    val dummyArray = Float32Array(arrayOf())
    var parent: TransformKt? = null
    var position: Float32Array = dummyArray
        get() {
            makeDirty()
            return field
        }
        set(value) {
            if (field == dummyArray) {
                field = value
            } else {
                field.set(value)
            }
            makeDirty()
        }
    var scale: Float32Array = dummyArray
        get() {
            makeDirty()
            return field
        }
        set(value) {
            if (field == dummyArray) {
                field = value
            } else {
                field.set(value)
            }
            makeDirty()
        }
    var localMatrix: Float32Array
    var worldMatrix: Float32Array
        get() {
            if (worldMatrixDirty) {
                val parrent = parent
                if (parrent == null) {
                    field.set(resolveLocalMatrix())
                } else {
                    mat4Multiply(field, parrent.worldMatrix, resolveLocalMatrix())
                }
                worldMatrixDirty = false
            }

            return field
        }
    var localMatrixDirty = true
    var worldMatrixDirty = true
    var children = mutableSetOf<TransformKt>()

    var orientation: Float32Array = dummyArray
        get() {
            makeDirty()
            return field
        }
        set(value) {
            if (field == dummyArray) {
                field = value
            } else {
                field.set(value)
            }
            makeDirty()
        }

    init {

        var buffer: ArrayBuffer
        var offset = 0

        if (options?.externalStorage != null) {
            buffer = options.externalStorage.buffer
            offset = options.externalStorage.offset
        } else {
            buffer = Float32Array(42).buffer
        }

        position = Float32Array(buffer, offset, 3)
        orientation = Float32Array(buffer, offset + 3 * Float32Array.BYTES_PER_ELEMENT, 4)
        scale = Float32Array(buffer, offset + 7 * Float32Array.BYTES_PER_ELEMENT, 3)
        localMatrix = Float32Array(buffer, offset + 10 * Float32Array.BYTES_PER_ELEMENT, 16)
        worldMatrix = Float32Array(buffer, offset + 26 * Float32Array.BYTES_PER_ELEMENT, 16)

        if (options != undefined) {
            if (options.position != undefined) {
                position.set(options.position as Array<Float>)
            }
            orientation.set(if (options.orientation != undefined) options.orientation as Float32Array else DEFAULT_ORIENTATION)
            scale.set(arrayOf(1f, 1f, 1f))

        }

    }

    fun getWorldPosition(out: Float32Array, position: Float32Array?) {
        if (position != null) {
            if (position != out) {
                out.set(position)
            }
        } else {
            MyVector3(0.0, 0.0, 0.0).into(out)
        }
        vec3TransformMat4(out, out, worldMatrix)
    }

    fun makeDirty(markLocalDirty: Boolean = true) {
        if (markLocalDirty) {
            localMatrixDirty = true
        }
        if (worldMatrixDirty) {
            return
        }
        worldMatrixDirty = true

        children.forEach { child ->
            child.makeDirty(false)
        }

    }

    fun addChild(transform: TransformKt) {

        children.add(transform)
        transform.parent = this
        transform.makeDirty(false)
    }

    fun resolveLocalMatrix(): Float32Array {
        val wasDirty = localMatrixDirty
        if (wasDirty) {
            mat4FromRotationTranslationScale(
                localMatrix,
                orientation,
                position,
                scale
            )
            localMatrixDirty = false
        }
        return localMatrix
    }
}
