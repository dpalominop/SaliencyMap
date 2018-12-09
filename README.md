# SaliencyMap

Nuestra atención se ve atraída por estímulos visualmente destacados. Es importante que los sistemas biológicos complejos detecten rápidamente presas, depredadores o parejas potenciales en un mundo visual desordenado. Sin embargo, la identificación simultánea de todos y cada uno de los objetivos interesantes en el campo visual tiene una complejidad computacional prohibitiva que la convierte en una tarea desalentadora incluso para los cerebros biológicos más sofisticados (Tsotsos, 1991), y mucho menos para cualquier computadora existente. Una solución, adoptada por los primates y muchos otros animales, es restringir el proceso de reconocimiento de objetos complejos a un área pequeña o unos pocos objetos al mismo tiempo.

# Requerimientos
## S.O.

Ubuntu 18.04

## Dependencias
Verificar que los utilitarios de compilación estén instalados:

```
sudo apt-get update
sudo apt-get install build-essential -y
```

Instalar OpenCV 3.4.2

se puede seguir este manual: [OpenCV](https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/)

# Compilar y Ejecutar

Moverse al directorio de descarga del repositorio:

```
cd ($DOWNLOAD_PATH)/SaliencyMap
```

Compilar:

```
make build
```

Ejecutar:

```
./out
```
