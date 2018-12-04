import os

this_dir = os.path.abspath(os.path.dirname(__file__))

c = get_config()

c.Exporter.template_file = os.path.join(this_dir, 'jupyter_template.tpl')
