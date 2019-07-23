---
layout: post
title:  "Les closures: comment ils fonctionnent en JS"
date:   2019-07-23 17:13:20
---

Il s'agit d'un mécanisme de gestion des fonctions particulier au langage JavaScript. Le point sur son fonctionnement.

Le langage JavaScript possède un mécanisme de gestion des fonctions particulier appelé closure. Les closures se basent sur des fonctions dites de première classe. Ce sont des fonctions qui peuvent être stockées dans des variables, envoyées dans d'autres fonctions ou retournées comme résultat d'une fonction.

#### Comment fonctionnent les closures ?

Si nous devions définir les closures en une phrase : une closure est un contexte crée par la déclaration d’une fonction, et qui permet à cette dite fonction d’accéder et manipuler des variables se trouvant en dehors de la portée de cette fonction. Ça va ? C’est assez clair ? Disons qu’une closure permet à une fonction foo d’accéder à toutes les variables et fonctions qui sont dans le contexte de déclaration (et non d’invocation) de cette fonction foo.

{% highlight js %}
var a = "Hello";
function foo(){
	console.log(a); // Hello
}
foo();
{% endhighlight js %}

Dans cet exemple, nous avons déclaré une variable jedi, et une fonction foo dans le même contexte — dans ce cas le contexte global. Lorsque nous exécutons la fonction  foo, celle-ci a bien accès à la variable  jedi. Je parie que vous avez écrit ce genre de code une dizaine de fois sans vous rendre compte que vous étiez en train de manipuler des closures !