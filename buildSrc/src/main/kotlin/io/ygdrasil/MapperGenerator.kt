package io.ygdrasil

import com.squareup.kotlinpoet.*
import kotlin.reflect.KClass


/**
 * This file is part of an experimental project that aims to generate code to support webgpu bindings.
 * Specifically, it seeks to create a mapper, a construct that can translate or "map" data from one representation to another.
 *
 */
val stringClass = String::class.toClassName()
val intClass = Int::class.toClassName()
val uIntClass = UInt::class.toClassName()

fun main() {

    val greeterClass = ClassName("", "Greeter")
    println(String::class.qualifiedName)
    generate(greeterClass, greeterClass, mapOf())
        .let { println(it) }
}

fun generate(input: ClassName, output: ClassName, propertyToMap:Map<ParameterSpec, ParameterSpec>): String {

    val test = ParameterSpec.builder("args", stringClass).build()
    //Int::class.to

    return FunSpec.builder("convert")
        .returns(output)
        .addParameter(test)
        .addStatement("%T(args).greet()", input)
        .build()
        .toString()
}

internal fun KClass<*>.toClassName() = ClassName(packageName, simpleClassName)

internal val KClass<*>.packageName
    get() = qualifiedName?.substringBeforeLast('.') ?: error("fail to get qualifiedName")
internal val KClass<*>.simpleClassName
    get() = qualifiedName?.substringAfterLast('.') ?: error("fail to get qualifiedName")