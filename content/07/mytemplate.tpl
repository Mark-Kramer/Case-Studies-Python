
{% extends 'full.tpl'%}
{% block any_cell %}
{% if 'Hard' in cell['metadata'].get('tags', []) %}
    <div style="border:thin solid red">
        {{ super() }}
    </div>
{% elif 'Medium' in cell['metadata'].get('tags', []) %}
    <div style="border:thin solid orange">
        {{ super() }}
    </div>
{% elif 'Easy' in cell['metadata'].get('tags', []) %}
    <div style="border:thin solid green">
        {{ super() }}
    </div>
{% else %}
    {{ super() }}
{% endif %}
{% endblock any_cell %}
