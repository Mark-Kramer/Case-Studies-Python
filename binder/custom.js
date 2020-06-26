// Include jquery in the head element
var script = document.createElement('script');
script.src = "http://ajax.googleapis.com/ajax/libs/jquery/1.3.2/jquery.min.js";
script.type = "text/javascript";
document.head.appendChild(script);

$( document ).ready(function() {
	var ext = "." + document.URL.split('.').pop().split(/\#|\?/)[0];
	var local = $('a[rel="local"]');

    // Update targets in links with rel=["external", "local"]
	$('a[rel="external"]').attr('target', '_blank');
	local.attr('target', '_blank');
	// Add link extension
	$.each(local, function( idx, a ) { 
		if ( a.href.search('#') >= 0 ) { a.href = a.href.split('#').join(ext); }
		else { a.href += ext };
	});
});
