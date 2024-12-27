Primero, es importante recordar que es bioconjugacion y como se relaciona con el termino electroforesis: 

La bioconjugación es el proceso de unir dos biomoléculas (por ejemplo, proteínas, ADN o enzimas) mediante enlaces químicos específicos para crear materiales híbridos con propiedades mejoradas.

Por otro lado, la electroforesis es una técnica utilizada para separar y analizar biomoléculas según su tamaño y carga eléctrica.

La relación entre ambas técnicas radica en que la electroforesis actúa como método de validación. Después de realizar una bioconjugación, es fundamental verificar si las moléculas se unieron correctamente.

En la investigación se identificó que los datos sobre electroforesis pueden encontrarse en dos formatos principales:
	1.	Imágenes (patrones visuales del gel).
	2.	Datos tabulares (información estructurada en filas y columnas).

Dependiendo del formato de los datos, se recomiendan diferentes modelos de IA para generar o analizar datos sintéticos.

1. Modelos para Imágenes:
	GANs (Generative Adversarial Networks):
	    •	Crean imágenes realistas simulando patrones visuales a partir de datos originales.
	    •	Utilizan dos redes neuronales (Generador y Discriminador) que compiten para producir datos sintéticos cada vez más precisos.
	    •	Son ideales para crear patrones visuales como bandas en imágenes de geles.
	VAEs (Variational Autoencoders):
	    •	Codifican los datos en un espacio latente (un mapa comprimido de características) y luego los reconstruyen.
	    •	Modelan distribuciones estadísticas, permitiendo generar datos más variados y controlados.
	    •	Son útiles para modificar características específicas en las imágenes.

2. Modelos para Datos Tabulares:
	TGAN (Tabular GAN):
	    •	Una versión optimizada de GAN para datos tabulares (filas y columnas).
	    •	Mantiene relaciones entre columnas, respetando características numéricas y categóricas.
	    •	Ideal para generar datos sintéticos manteniendo patrones complejos en tablas.
	VAE Tabular:
	    •	Una variación de VAE especializada en datos tabulares.
	    •	Aprende distribuciones estadísticas de los datos y genera filas nuevas manteniendo la estructura.
	    •	Ideal para patrones estadísticos precisos y análisis predictivos.

Con respecto a la seleccion de modelo segun el formato de datos deseado.
	Imágenes:
	    •	Usar GAN para generar patrones aleatorios y realistas.
	    •	Usar VAE para controlar variaciones y crear datos estadísticamente coherentes.
	Datos Tabulares:
	    •	Usar TGAN para generar patrones tabulares aleatorios.
	    •	Usar VAE Tabular para generar datos más precisos y estructurados.

Finalmente, Debido a que se está trabajando con cientificos y que los datos experimentales son costosos o difíciles de obtener; se optó por el uso de GANs, debido a que se se busca generar datos muy similares a los datos reales y explorar diferentes distribuciones como patrones. Aun así esta decision, podrá variar en base a los datos encontrados y si se necesita filtrar los resultados, que gracias a VAE, se podria seleccionar los datos mas utiles o relevantes.

