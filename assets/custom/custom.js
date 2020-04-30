// Put your custom javascript here


var script = document.createElement('script');

script.src = "https://rawgit.com/showdownjs/showdown/develop/dist/showdown.min.js";
script.type = "text/javascript";

document.head.appendChild(script);

function md2html() {
	var converter = new showdown.Converter();
	var classes = ["question", "python-note", "warning", "math-note"];

	c = 0;
	while (classes[c]) {
		ii = 0;
		X = document.getElementsByClassName(classes[c]);
		while (X[ii]) {
			X[ii].innerHTML = converter.makeHtml(X[ii].innerHTML);
			ii++;
		}
		c++;
	}
}

document.addEventListener('turbolinks:load', () => {
  md2html()
})

initFunction(md2html)