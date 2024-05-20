package io.ygdrasil

import com.squareup.kotlinpoet.*
import kotlin.reflect.KClass


/**
 * This file is part of an experimental project that aims to generate code to support webgpu bindings.
 * Specifically, it seeks to create a mapper, a construct that can translate or "map" data from one representation to another.
 *
 */
val stringClass = String::class.toTypeRef()
val intClass = Int::class.toTypeRef()
val uIntClass = UInt::class.toTypeRef()
val uLongClass = ULong::class.toTypeRef()
val memScopeClass: TypeRef = ClassRef("kotlinx.cinterop", "MemScope")
val webgpuPackage = "io.ygdrasil.wgpu"
val kotlinPackage = "kotlin"

fun main() {

    /*
    data class ShaderModuleDescriptor(
	var code: String,
	var label: String? = null,
	var sourceMap: Any? = null,
	var compilationHints: Array<CompilationHint>? = null
) {
	data class CompilationHint(
		var entryPoint: String,
		// TODO
		//var layout: dynamic /* GPUPipelineLayout? | "auto" */
	)
}

public final expect class WGPUShaderModuleDescriptor public constructor(rawPtr: kotlinx.cinterop.NativePtr ) : kotlinx.cinterop.CStructVar {
    @kotlin.Deprecated public expect companion object : kotlinx.cinterop.CStructVar.Type {
    }

    public expect final var hintCount: platform.posix.size_t /* = kotlin.ULong */ /* compiled code */

    public expect final var hints: kotlinx.cinterop.CPointer<webgpu.WGPUShaderModuleCompilationHint>?

    public expect final var label: kotlinx.cinterop.CPointer<kotlinx.cinterop.ByteVar /* = kotlinx.cinterop.ByteVarOf<kotlin.Byte> */>?

    public expect final var nextInChain: kotlinx.cinterop.CPointer<webgpu.WGPUChainedStruct>?
}
     */
    val compilationHintsClass = GenericClassRef(kotlinPackage, "List", ClassRef(webgpuPackage, "ShaderModuleDescriptor.CompilationHint"))

    val input = ClassRef(webgpuPackage, "ShaderModuleDescriptor")
    val output = ClassRef("webgpu", "WGPUShaderModuleDescriptor")

    generate(
        input,
        output,
        listOf(
            PropertyMap(stringClass.toProperty("code", true), stringClass.toProperty("code", true)),
            PropertyMap(stringClass.toProperty("label", true), stringClass.toProperty("label", true)),
            PropertyMap(
                compilationHintsClass.toProperty("compilationHints", true),
                uLongClass.toProperty("hintCount", false)
            )
        ),
        receiver = memScopeClass
    ).let { println(it) }
}

fun generate(
    input: TypeRef,
    output: TypeRef,
    propertyToMap: List<PropertyMap>,
    receiver: TypeRef? = null
): String {
    return FunSpec.builder("convert")
        .addParameter(input.toParameterSpec("input"))
        .receiver(receiver)
        .returns(output.toClassName())
        .apply {
            addStatement("val output = alloc<${output.canonicalName}> {")
            propertyToMap.forEach { with(it) { mapProperty() } }
            addStatement("}")
            addStatement("return output")
        }
        .build()
        .toString()
}

private fun TypeRef.toParameterSpec(name: String): ParameterSpec {
    return ParameterSpec.builder(name, toClassName()).build()
}

data class PropertyMap(val source: Property, val destination: Property) {

    data class Property(val type: TypeRef, val name: String, val isNullable: Boolean) {

        internal fun isList(): Boolean {
            return type is GenericClassRef && type.className == "List"
        }

        internal fun isNumeric(): Boolean {
            return type is ClassRef && type.packageName == kotlinPackage && type.className in listOf("Int", "Long", "UInt", "ULong")
        }
    }

    internal fun FunSpec.Builder.mapProperty() {
        when {
            source.type == destination.type -> addStatement("\t${destination.name} = input.${source.name}")
            source.isList() && destination.isNumeric() -> addStatement("\t${destination.name} = input.${source.name}.size.toULong()")
            else -> error("not supported")
        }
    }


}


internal fun TypeRef.toProperty(name: String, isNullable: Boolean = false): PropertyMap.Property {
    return PropertyMap.Property(this, name, isNullable)
}

internal fun FunSpec.Builder.receiver(receiver: TypeRef?) = apply {
    if (receiver != null) receiver(receiver.toClassName())
}

internal fun ClassName.toProperty(argName: String, isNullable: Boolean = false): ParameterSpec {
    return ParameterSpec.builder(argName, this.copy(nullable = isNullable)).build()
}

internal fun KClass<*>.toTypeRef(): TypeRef = ClassRef(packageName, simpleClassName)

internal val KClass<*>.packageName
    get() = qualifiedName?.substringBeforeLast('.') ?: error("fail to get qualifiedName")
internal val KClass<*>.simpleClassName
    get() = qualifiedName?.substringAfterLast('.') ?: error("fail to get qualifiedName")