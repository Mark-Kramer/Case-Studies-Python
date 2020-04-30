// Put your custom javascript here
// <script type="text/javascript" src="https://rawgit.com/showdownjs/showdown/develop/dist/showdown.min.js"></script>

function load_script() {
	var converter = new showdown.Converter();

	ii = 0;
	X = document.getElementsByClassName("question");
	while (X[ii]) {
		X[ii].innerHTML = converter.makeHtml(X[ii].innerText);
		// X[ii].innerHTML = "Hello";
		ii++;
	}
}