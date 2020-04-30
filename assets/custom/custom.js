// Put your custom javascript here
// <script type="text/javascript" src="https://rawgit.com/showdownjs/showdown/develop/dist/showdown.min.js"></script>


var script = document.createElement('script');

script.src = "https://rawgit.com/showdownjs/showdown/develop/dist/showdown.min.js";
script.type = "text/javascript";

document.head.appendChild(script);

function load_script() {
	var converter = new showdown.Converter();
	var classes = ["question", "python-note", "warning", "math-note"];
	c = 0;

	while (classes[c]) {
		ii = 0;
		X = document.getElementsByClassName(classes[c]);
		while (X[ii]) {
			X[ii].innerHTML = converter.makeHtml(X[ii].innerHTML);
			// X[ii].innerHTML = "Hello";
			ii++;
		}
		c++;
	}
}

document.addEventListener('turbolinks:load', () => {
  load_script()
})
// initFunction(load_script);