import json
import os
import os.path
import re
import urllib
import requests
import gradio as gr # type: ignore
from modules import script_callbacks, sd_models, shared, sd_hijack, images, scripts # type: ignore
current_extension_directory = scripts.basedir()
from PIL import Image
import base64
import csv
from io import BytesIO
from lxml.html.clean import Cleaner

import importlib.util

def import_lora_module():
	# import/update the lora module if its available
	try:
		spec = importlib.util.find_spec('extensions.sd-webui-additional-networks.scripts.model_util')
		if spec:
			additional_networks = importlib.util.module_from_spec(spec)
			spec.loader.exec_module(additional_networks)
		else:
			additional_networks = None
	except:
		additional_networks = None
	return additional_networks

def import_lora_module_builtin():
	# import/update the lora module if its available from the builtin extensions
	try:
		spec = importlib.util.find_spec('extensions-builtin.Lora.lora')
		if spec:
			additional_networks_builtin = importlib.util.module_from_spec(spec)
			spec.loader.exec_module(additional_networks_builtin)
		else:
			additional_networks_builtin = None
	except:
		additional_networks_builtin = None
	return additional_networks_builtin


def import_lycoris_module():
	# import/update the lycoris module if it's available

	possible_lycoris_modules = [
		'extensions-builtin.a1111-sd-webui-lycoris.lycoris',
		'extensions.a1111-sd-webui-lycoris.lycoris'
	]
	for module in possible_lycoris_modules:
		try:
			spec = importlib.util.find_spec(module)
			if spec:
				loaded_lycoris_module = importlib.util.module_from_spec(spec)
				spec.loader.exec_module(loaded_lycoris_module)
				return loaded_lycoris_module
		except:
			pass

	return None

# define a global Cleaner instance with specific options for sanitization
cleaner = Cleaner(
    safe_attrs_only=True,  # Only allow safe attributes
    host_whitelist=set(['www.youtube.com'])
)

def sanitize_html(html_content):
    # check if the HTML content is empty, if so we dont have to do anything, return an empty string
    if html_content is None or not html_content.strip():
        return ""

    # check if the entire HTML string is surrounded in comment tags (This is done by some civitai extensions)
    if html_content.strip().startswith('<!--') and html_content.strip().endswith('-->'):
        # Remove the comment tags
        html_content = html_content.strip()[4:-3].strip()

    try:
        # clean the HTML content using the global Cleaner instance
        cleaned_html = cleaner.clean_html(html_content)
    except Exception as e:
        # if there is an error cleaning the HTML, return "Unable to parse HTML"
        return "Unable to parse HTML"

	# return the cleaned HTML
    return cleaned_html

# try and get the lora module
additional_networks = import_lora_module()
additional_networks_builtin = import_lora_module_builtin()

# try and get the lycoris module
lycoris_module = import_lycoris_module()

refresh_symbol = '🔄'
update_symbol = '↙️'

html_ext_pattern = r'html'
civitai_ext_pattern = r'civitai.info'
md_ext_pattern = r'md'
txt_ext_pattern = r'txt'
tags_ext_pattern = r'tags'
prompts_ext_pattern = r'(?:prompt|prompts)'
img_ext_pattern = r'(?:png|jpg|jpeg|webp|jxk|avif)'
all_ext_pattern = r'(?:' + html_ext_pattern\
				  + r'|' + civitai_ext_pattern\
				  + r'|' + md_ext_pattern\
				  + r'|' + txt_ext_pattern\
				  + r'|' + tags_ext_pattern\
				  + r'|' + prompts_ext_pattern\
				  + r'|' + img_ext_pattern\
				  + r')'

def is_in_directory(parent_dir, child_path):
	# get the directory of the child path
	child_dir = os.path.dirname(child_path)

	# get the absolute paths of both directories
	parent_dir = os.path.abspath(os.path.realpath(parent_dir))
	child_dir = os.path.abspath(os.path.realpath(child_dir))

	# return false if either directory is not a valid directory
	if not os.path.isdir(parent_dir) or not os.path.isdir(child_dir):
		return False

	# get the common prefix of the paths to see if the child dir is in the parent
	common_prefix = os.path.commonprefix([parent_dir, child_dir])
	return common_prefix == parent_dir and child_dir != parent_dir

def is_dir_in_list(dir_list, check_dir):
	# Convert all directories in the list to absolute paths
	dir_list = [os.path.abspath(d) for d in dir_list]
	# Convert the specified directory to an absolute path
	check_dir = os.path.abspath(check_dir)
	# Check if the specified directory is in the list of directories
	for dir_path in dir_list:
		if os.path.samefile(check_dir, dir_path):
			return True
	return False

def natural_order_number(s):
	# split a string into segments of strings and ints that will be used to sort naturally
	return [int(x) if x.isdigit() else x.lower() for x in re.split('(\d+)', s)]

def clean_modelname(modelname):
	# remove the extension and the hash if it exists at the end of the model name (this is added by a1111) and
	# if the model name contains a path (which happens when a checkpoint is in a subdirectory) just return the model name portion
	return re.sub(r"(?i)(\.pt|\.bin|\.ckpt|\.safetensors)?( \[[a-f0-9]{10,12}\]|\([a-f0-9]{10,12}\))?$", "", modelname).split("\\")[-1].split("/")[-1]

# keep a copy of the choices to give control to user when to refresh
checkpoint_choices = []
embedding_choices = []
hypernetwork_choices = []
lora_choices = []
lycoris_choices = []
tags = {
	"checkpoints": {},
	"embeddings": {},
	"hypernetworks": {},
	"loras": {},
	"lycoris": {}
}

def search_for_tags(model_names, model_tags, paths):
	model_tags.clear()
	general_tag_pattern = re.compile(r'^.*(?i:\.tags)$')

	# support the ability to check multiple paths
	for path in paths:
		# loop through all files in the path and any subdirectories
		for dirpath, dirnames, filenames in os.walk(path, followlinks=True):
			# get a list of all parent directories
			directories = dirpath.split(os.path.sep)

			index_models = []
			if shared.opts.model_preview_xd_name_matching == "Index":
				index_txt_filename = next((filename for filename in filenames if filename.lower() == "index.txt"), None)
				if index_txt_filename is not None:
					index_txt_path = os.path.join(dirpath, index_txt_filename)
					output_text = ""
					with open(index_txt_path, "r", encoding="utf8") as file:
						output_text = file.read()
					index_models = [model.strip() for model in output_text.replace(",", "\n").splitlines()]

			# check each file to see if it is a preview file
			for filename in filenames:
				file_path = os.path.join(dirpath, filename)
				if general_tag_pattern.match(filename):
					for model_name in model_names:
						clean_model_name = clean_modelname(model_name)

						# if we are not using folder match mode look for files normally otherwise we are using folder match mode so make sure at least one parent directory is equal to the name of the model
						if shared.opts.model_preview_xd_name_matching == "Folder" and clean_model_name not in directories:
							continue

						index_has_model = False
						index_models_pattern = None
						if shared.opts.model_preview_xd_name_matching == "Index":
							index_has_model = clean_model_name in index_models
							filtered_index_models = [re.escape(model) for model in index_models if model != clean_model_name]
							if len(filtered_index_models) > 0:
								index_models_pattern = re.compile(r'^(?:' + r'|'.join(filtered_index_models) + r')(?i:\.' + tags_ext_pattern + r')$')
								if index_models_pattern.match(filename):
									continue

						if shared.opts.model_preview_xd_name_matching == "Strict" or (not index_has_model and shared.opts.model_preview_xd_name_matching == "Index"):
							tag_pattern = re.compile(r'^' + re.escape(clean_model_name) + r'(?i:\.' + tags_ext_pattern + r')$')
						elif shared.opts.model_preview_xd_name_matching == "Folder" or (index_has_model and shared.opts.model_preview_xd_name_matching == "Index"):
							tag_pattern = re.compile(r'^.*(?i:\.' + tags_ext_pattern + r')$')
						else:
							tag_pattern = re.compile(r'^.*' + re.escape(clean_model_name) + r'.*(?i:\.' + tags_ext_pattern + r')$')

						if tag_pattern.match(filename):
							output_text = ""
							with open(file_path, "r", encoding="utf8") as file:
								output_text = file.read()
							if output_text.strip() != "":
								if model_name in model_tags:
									model_tags[model_name] += f", {output_text}"
								else:
									model_tags[model_name] = output_text

def list_all_models():
	global checkpoint_choices
	# gets the list of checkpoints
	model_list = sd_models.checkpoint_tiles()
	checkpoint_choices = sorted(model_list, key=natural_order_number)
	search_for_tags(checkpoint_choices, tags["checkpoints"], get_checkpoints_dirs())
	return checkpoint_choices

def list_all_embeddings():
	global embedding_choices
	# Embeddings may not have been loaded yet. (Fixes empty embeddings list on startup) -n15g
	sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
	# get the list of embeddings
	list = [x for x in sd_hijack.model_hijack.embedding_db.word_embeddings.keys()]
	list.extend([x for x in sd_hijack.model_hijack.embedding_db.skipped_embeddings.keys()])
	embedding_choices = sorted(list, key=natural_order_number)
	search_for_tags(embedding_choices, tags["embeddings"], get_embedding_dirs())
	return embedding_choices

def list_all_hypernetworks():
	global hypernetwork_choices
	# get the list of hyperlinks
	list = [x for x in shared.hypernetworks.keys()]
	hypernetwork_choices = sorted(list, key=natural_order_number)
	search_for_tags(hypernetwork_choices, tags["hypernetworks"], get_hypernetwork_dirs())
	return hypernetwork_choices

def list_all_loras():
	global lora_choices, additional_networks, additional_networks_builtin
	# create an empty set for lora models
	loras = set()

	# import/update the lora module
	additional_networks = import_lora_module()
	if additional_networks is not None:
		# copy the list of models
		loras_list = additional_networks.lora_models.copy()
		# remove the None item from the list
		loras_list.pop("None", None)
		# remove hash from model
		loras_list = [re.sub(r'\([a-fA-F0-9]{10,12}\)$', '', model) for model in loras_list.keys()]
		loras.update(loras_list)

	# import/update the builtin lora module
	additional_networks_builtin = import_lora_module_builtin()
	if additional_networks_builtin is not None:
		# copy the list of models
		loras_list = additional_networks_builtin.available_loras.copy()
		# remove the None item from the list
		loras_list.pop("None", None)
		loras.update(loras_list.keys())

	# return the list
	lora_choices = sorted(loras, key=natural_order_number)
	search_for_tags(lora_choices, tags["loras"], get_lora_dirs())
	return lora_choices

def list_all_lycorii():
	global lycoris_choices, lycoris_module
	# create an empty set for lycoris models
	lycorii = set()

	# import/update the lycoris module
	lycoris_module = import_lycoris_module()
	if lycoris_module is not None:
		# copy the list of models
		lycorii_list = lycoris_module.available_lycos.copy()
		# remove the None item from the list
		lycorii_list.pop("None", None)
		# remove hash from model
		lycorii_list = [re.sub(r'\([a-fA-F0-9]{10,12}\)$', '', lyco) for lyco in lycorii_list.keys()]
		lycorii.update(lycorii_list)

	# return the list
	lycoris_choices = sorted(lycorii, key=natural_order_number)
	search_for_tags(lycoris_choices, tags["lycoris"], get_lycoris_dirs())
	return lycoris_choices

def refresh_models(choice = None, filter = None):
	global checkpoint_choices
	# update the choices for the checkpoint list
	checkpoint_choices = list_all_models()
	return filter_models(filter), *show_model_preview(choice)

def refresh_embeddings(choice = None, filter = None):
	global embedding_choices
	# update the choices for the embeddings list
	embedding_choices = list_all_embeddings()
	return filter_embeddings(filter), *show_embedding_preview(choice)

def refresh_hypernetworks(choice = None, filter = None):
	global hypernetwork_choices
	# update the choices for the hypernetworks list
	hypernetwork_choices = list_all_hypernetworks()
	return filter_hypernetworks(filter), *show_hypernetwork_preview(choice)

def refresh_loras(choice = None, filter = None):
	global lora_choices
	# update the choices for the lora list
	lora_choices = list_all_loras()
	return filter_loras(filter), *show_lora_preview(choice)

def refresh_lycorii(choice = None, filter = None):
	global lycoris_choices
	# update the choices for the lycoris list
	lycoris_choices = list_all_lycorii()
	return filter_lycorii(filter), *show_lycoris_preview(choice)

def filter_choices(choices, filter, tags_obj):
	filtered_choices = choices
	if filter is not None and filter.strip() != "":
		# filter the choices based on the provided filter string
		filter_tags = [tag.strip().lower() for tag in filter.split(",")]
		filtered_choices = [choice for choice in filtered_choices if
							all(tag in tags_obj.get(choice, '').lower() for tag in filter_tags) or
							all(tag in choice.lower() for tag in filter_tags)]
	return filtered_choices

def filter_models(filter=None):
	filtered_checkpoint_choices = filter_choices(checkpoint_choices, filter, tags["checkpoints"])
	return gr.Dropdown.update(choices=filtered_checkpoint_choices)

def filter_embeddings(filter=None):
	filtered_embedding_choices = filter_choices(embedding_choices, filter, tags["embeddings"])
	return gr.Dropdown.update(choices=filtered_embedding_choices)

def filter_hypernetworks(filter=None):
	filtered_hypernetwork_choices = filter_choices(hypernetwork_choices, filter, tags["hypernetworks"])
	return gr.Dropdown.update(choices=filtered_hypernetwork_choices)

def filter_loras(filter=None):
	filtered_lora_choices = filter_choices(lora_choices, filter, tags["loras"])
	return gr.Dropdown.update(choices=filtered_lora_choices)

def filter_lycorii(filter=None):
	filtered_lycoris_choices = filter_choices(lycoris_choices, filter, tags["lycoris"])
	return gr.Dropdown.update(choices=filtered_lycoris_choices)

def update_checkpoint(name):
	# update the selected preview for checkpoint tab
	new_choice = find_choice(checkpoint_choices, name)
	return new_choice, *show_model_preview(new_choice)

def update_embedding(name):
	# update the selected preview for embedding tab
	new_choice = find_choice(embedding_choices, name)
	return new_choice, *show_embedding_preview(new_choice)

def update_hypernetwork(name):
	# update the selected preview for hypernetwork tab
	new_choice = find_choice(hypernetwork_choices, name)
	return new_choice, *show_hypernetwork_preview(new_choice)

def update_lora(name):
	# update the selected preview for lora tab
	new_choice = find_choice(lora_choices, name)
	return new_choice, *show_lora_preview(new_choice)

def update_lycorii(name):
	# update the selected preview for LyCORIS tab
	new_choice = find_choice(lycoris_choices, name)
	return new_choice, *show_lycoris_preview(new_choice)

def find_choice(list, name):
	# clean the name from the list and match a choice to the model
	# TODO there could be name collisions here that may need to be handled in the future.
	for choice in list:
		cleaned_name = clean_modelname(choice)
		if cleaned_name == name:
			return choice
	return name

def create_html_iframe(file, is_in_a1111_dir):
	if is_in_a1111_dir:
		# escape special URL characters from the filename
		encoded_file_path = urllib.parse.quote(file, safe='/:\\')
		# create the iframe html code
		html_code = f'<iframe src="file={encoded_file_path}"></iframe>'
	else:
		html_code = ""
		# the html file isnt located in the a1111 directory so load the html file as a base64 string instead of linking to it
		with open(file, 'rb') as html_file:
			html_data = base64.b64encode(html_file.read()).decode()
			html_code = f'<iframe src="data:text/html;charset=UTF-8;base64,{html_data}"></iframe>'
	return html_code

def extract_civitai_image_key(url):
    pattern = r"https?://(?:image(?:cache)?\.civitai\.com)/xG1nkqKTMzGDvpLrqFT7WA/([a-f0-9-]+)"
    match = re.match(pattern, url)
    if match:
        return match.group(1)
    return None

def convert_image_to_base64(url):
	# Only cache the image if setting is on
	if not shared.opts.model_preview_xd_cache_images_civitai_info:
		return url

	# Get the image key from the URL
	image_key = extract_civitai_image_key(url)

	if image_key is None:
		# Can't find the image key, the given url isn't in an expected form, just return the input URL
		return url

	# Construct the cache directory path
	cache_directory = os.path.join(current_extension_directory, 'civit_cache')
	
	# Check if the cache directory exists, if not, create it
	if not os.path.exists(cache_directory):
		os.makedirs(cache_directory)
	
	# Construct the image path within the cache directory
	image_path = os.path.join(cache_directory, image_key)
	
	# Check if the image file already exists in the cache directory
	if os.path.isfile(image_path):
		# If it exists, read the base64 data from the file
		with open(image_path, "r") as f:
			base64_data_uri = f.read()
		f.close()
		# Return the cached uri
		return base64_data_uri
	else:
		# If the image file doesn't exist in the cache directory, download it
		response = requests.get(url)

		# Check if the request was successful
		if response.status_code == 200:
			image_data = response.content
		else:
			# If not successful, return the input URL
			return url

		try:
			# Attempt to open the image using PIL
			image = Image.open(BytesIO(image_data))
			# Encode the image data to base64
			base64_image = base64.b64encode(image_data).decode('utf-8')
		except Image.UnidentifiedImageError:
			# If the image format is not recognized, return the input URL
			return url

		# Determine the image format
		if image.format:
			image_format = image.format
		else:
			image_format = "PNG"

		# Construct the base64 data URI
		base64_data_uri = f"data:image/{image_format};base64,{base64_image}"

		# Write the base64 data to a file in the cache directory
		with open(image_path, "w") as f:
			f.write(base64_data_uri)
		f.close

		print(f"SD Model Preview caching image {image_path}")

		return base64_data_uri

def create_civitai_info_html(file):
	# initialize the info object
	data = {}

	# read the civitai.info file
	if os.path.isfile(file):
		with open(file, 'r') as f:
			data = json.load(f)
		f.close()

	# Sanitize the HTML content of the description properties
	data['description'] = sanitize_html(data.get('description', ''))
	if 'model' in data:
		data['model']['description'] = sanitize_html(data['model'].get('description', ''))

	# build the html
	civitai_info_html = [f"""<div class='civitai-info'>
	<h1 id="ci-name">{data.get('name','')}</h1>
	<ul>
    <li><strong>ID:</strong> <span id="ci-id">{data.get('id','')}</span></li>
    <li><strong>Model ID:</strong> <a id="ci-modelId" href="https://civitai.com/models/{data.get('modelId','')}" target="_blank">{data.get('modelId','')}</a></li>
    <li><strong>Created At:</strong> <span id="ci-createdAt">{data.get('createdAt','')}</span></li>
    <li><strong>Updated At:</strong> <span id="ci-updatedAt">{data.get('updatedAt','')}</span></li>
    <li><strong>Base Model:</strong> <span id="ci-baseModel">{data.get('baseModel','')}</span></li>
	<li><strong>Trained Words:</strong> <span id="ci-trainedWords">{"None Specified" if (not data.get('trainedWords',None) or len(data.get('trainedWords',[])) == 0) else "<ul><li>" + "</li><li>".join(data.get('trainedWords',[])) + "</li></ul>"}</span></li>
    <li><strong>Early Access Time Frame:</strong> <span id="ci-earlyAccessTimeFrame">{data.get('earlyAccessTimeFrame','')}</span></li>
    </ul>
	<details open>
		<summary><strong>Description:</strong></summary>
    	<div id="ci-description" class="description">{data.get('description','')}</div>
	</details>
	<details>
  		<summary><strong>Stats:</strong></summary>
		<ul>
			<li><strong>Download Count:</strong> <span id="ci-downloadCount">{data.get('stats',{}).get('downloadCount','')}</span></li>
			<li><strong>Rating Count:</strong> <span id="ci-ratingCount"{data.get('stats',{}).get('ratingCount','')}></span></li>
			<li><strong>Rating:</strong> <span id="ci-rating">{data.get('stats',{}).get('rating','')}</span></li>
		</ul>
	</details>
	<details>
  		<summary><strong>Model Information:</strong></summary>
		<ul>
			<li><strong>Name:</strong> <span id="ci-modelName">{data.get('model',{}).get('name','')}</span></li>
			<li><strong>Type:</strong> <span id="ci-modelType">{data.get('model',{}).get('type','')}</span></li>
			<li><strong>NSFW:</strong> <span id="ci-modelNsfw">{data.get('model',{}).get('nsfw','')}</span></li>
			<li><strong>POI:</strong> <span id="ci-modelPoi">{data.get('model',{}).get('poi','')}</span></li>
			<li><strong>Description:</strong> <div id="ci-modelDescription" class="description">{data.get('model',{}).get('description','')}</div></li>
		</ul>
	</details>
	<details>
		<summary><strong>Files:</strong></summary>
	"""]

	for i, data_file in enumerate(data.get('files',[])):
		civitai_info_html.append(f"""<details>
			<summary><strong id="ci-fileName-{i}">{data_file.get('name','')}</strong></summary>
			<ul>
				<li><strong>ID:</strong> <span id="ci-fileId-{i}">{data_file.get('id','')}</span></li>
				<li><strong>Size (KB):</strong> <span id="ci-fileSizeKB-{i}">{data_file.get('sizeKB','')}</span></li>
				<li><strong>Type:</strong> <span id="ci-fileType-{i}">{data_file.get('type','')}</span></li>
				<li><strong>Format:</strong> <span id="ci-fileFormat-{i}">{data_file.get('metadata',{}).get('format','')}</span></li>
				<li><strong>Fp:</strong> <span id="ci-fileFp-{i}">{data_file.get('metadata',{}).get('fp','')}</span></li>
				<li><strong>Size:</strong> <span id="ci-fileSize-{i}">{data_file.get('metadata',{}).get('size','')}</span></li>
				<li><strong>Pickle Scan Result:</strong> <span id="ci-pickleScanResult-{i}">{data_file.get('pickleScanResult','')}</span></li>
				<li><strong>Pickle Scan Message:</strong> <span id="ci-pickleScanMessage-{i}">{data_file.get('pickleScanMessage','')}</span></li>
				<li><strong>Virus Scan Result:</strong> <span id="ci-virusScanResult-{i}">{data_file.get('virusScanResult','')}</span></li>
				<li><strong>Scanned At:</strong> <span id="ci-scannedAt-{i}">{data_file.get('scannedAt','')}</span></li>
				<li><strong>Download URL:</strong> <a id="ci-downloadUrl-{i}" href="{data_file.get('downloadUrl','')}" target="_blank">{data_file.get('downloadUrl','')}</a></li>
			</ul>
		</details>
		""")

	civitai_info_html.append("""</details>
		<br>
		<div id="ci-images" class="img-container-set">
		""")
	
	for i, image in enumerate(data.get('images',[])):

		# Get the meta data object from the image
		meta_data = image.get('meta', None)

		# Initialize html meta list as not found incase its empty
		meta_list_items = "<li>No Meta Data Found</li>"

		image_url = convert_image_to_base64(image.get('url',''))

		civitai_info_html.append(f"""<div class='img-prop-container'><div class='img-container'>
			<img id="ci-image-{i}" src="{image_url}" onclick="imageZoomIn(event)" />
			""")

		# if there is prompt/meta data
		if meta_data:
			# Create the HTML list of all the meta data keys
			meta_list_items = "\n".join([f"<li><strong>{key}:</strong> {meta_data.get(key,'')}</li>" for key in meta_data])

			# Build the meta data string that will be copied when you press the copy button
			meta_tags = list(meta_data.keys())
			meta_out = []
			if "prompt" in meta_tags:
				meta_out.append(f"{image['meta']['prompt']}\n")
			if "negativePrompt" in meta_tags:
				meta_out.append(f"Negative prompt: {image['meta']['negativePrompt']}\n")
			for i, tag in enumerate(meta_tags):
				if tag == "cfgScale":
					# Add the cfgScale meta data to the output string
					meta_out.append(f"CFG scale: {image['meta']['cfgScale']}, ")
				elif tag != "prompt" and tag != "negativePrompt" and tag != "resources" and tag != "hashes":
					# Add the other meta data to the output string, convert the tag to Proper case
					meta_out.append(re.sub(r'\b\w', lambda x: x.group(0).upper(), tag, count=1) + ": " + str(image['meta'][tag]) + ", ")
			# Remove any trailing commas or whitespace
			meta_out_string = "".join(meta_out).rstrip(", ")

			# Add the button and an invisible textarea that will let you copy the meta data as a prompt
			if meta_out_string.strip() != "":
				civitai_info_html.append('<div class="img-meta-ico" title="Copy Metadata" onclick="metaDataCopy(event)"></div>')
				civitai_info_html.append(f'<textarea class="img-meta">{meta_out_string}</textarea>')

		civitai_info_html.append(f"""</div>
			<details class='img-properties-list'>
				<summary><strong>Properties:</strong></summary>
				<ul>
					<li><strong>URL:</strong> <a id="ci-image-URL-{i}" href="{image.get('url','')}" target="_blank">{image.get('url','')}</a></li>
					<li><strong>NSFW:</strong> <span id="ci-image-nsfw-{i}">{image.get('nsfw','')}</span></li>
					<li><strong>Meta:</strong>
						<ul id="ci-image-meta-{i}">
							{meta_list_items}
						</ul>
					</li>
				</ul>
			</details>
		</div>
		""")

	civitai_info_html.append("</div></div>")
	return "".join(civitai_info_html)

def create_html_img(file, is_in_a1111_dir):
	# create the html to display an image along with its meta data
	image = Image.open(file)
	# load the image to memory (needed for getting the meta data)
	image.load()
	# get the prompt data
	metadata, _ = images.read_info_from_image(image)

	# set default order to 0
	order = 0
	# if strict naming is on, search the file name for a number at the end of the file and use that for its order
	if shared.opts.model_preview_xd_name_matching == "Strict":
		# get the file name without extension
		file_name, file_extension = os.path.splitext(os.path.basename(file))
		# search for '{anything}.{number}' in the file name and return the number
		image_number = re.search(".*\.(\d+)$", file_name)
		order = int(image_number.group(1)) if image_number else 0

	if is_in_a1111_dir:
		# escape special URL characters from the filename
		encoded_file_path = urllib.parse.quote(file, safe='/:\\')
		# create the html for the image
		html_code = f'<div class="img-container" style="order:{order}"><img src=file={encoded_file_path} onclick="imageZoomIn(event)" />'
	else:
		# linking to the image wont work so convert it to a base64 byte string
		with open(file, "rb") as img_file:
			img_data = base64.b64encode(img_file.read()).decode()
		# create the html for the image
		html_code = f'<div class="img-container" style="order:{order}"><img src="data:image/{image.format};base64,{img_data}" onclick="imageZoomIn(event)" />'

	# if the image has prompt data in the meta data also add some elements to support copying the prompt to clipboard
	if metadata is not None and metadata.strip() != "":
		html_code += '<div class="img-meta-ico" title="Copy Metadata" onclick="metaDataCopy(event)"></div>'
		html_code += f'<textarea class="img-meta">{metadata}</textarea>'
	html_code += "</div>\n"
	# return the html code
	return html_code

def search_and_display_previews(model_name, paths):
	html_generic_pattern = re.compile(r'^.*(?i:\.' + html_ext_pattern + r')$')
	civitai_generic_pattern = re.compile(r'^.*(?i:\.' + civitai_ext_pattern + r')$')
	md_generic_pattern = re.compile(r'^.*(?i:\.' + md_ext_pattern + r')$')
	txt_generic_pattern = re.compile(r'^.*(?i:\.' + txt_ext_pattern + r')$')
	prompts_generic_pattern = re.compile(r'^.*(?i:\.' + prompts_ext_pattern + r')$')
	img_generic_pattern = re.compile(r'^.*(?i:\.' + img_ext_pattern + r')$')
	# create patters for the supported preview file types
	# `model_name` will be the name of the model to check for preview files for
	if shared.opts.model_preview_xd_name_matching == "Strict" or shared.opts.model_preview_xd_name_matching == "Index":
		# strict naming is intended to avoid name collision between 'checkpoint1000' and 'checkpoint10000'.
		# Using a loose naming rule preview files for 'checkpoint10000' would show up for 'checkpoint1000'
		# The rules for strict naming are:
		# HTML previews should follow {model}.html example 'checkpoint1000.html'
		html_pattern = re.compile(r'^' + re.escape(model_name) + r'(?i:\.' + html_ext_pattern+ r')$')
		# Civitai info files should follow {model}.civitai.info
		civitai_pattern = re.compile(r'^' + re.escape(model_name) + r'(?i:\.' + civitai_ext_pattern+ r')$')
		# Markdown previews should follow {model}.md example 'checkpoint1000.md'
		md_pattern = re.compile(r'^' + re.escape(model_name) + r'(?i:\.' + md_ext_pattern+ r')$')
		# Prompt lists should follow {model}.prompt example 'checkpoint1000.prompt'
		prompts_pattern = re.compile(r'^' + re.escape(model_name) + r'(?i:\.' + prompts_ext_pattern + r')$')
		# Text files previews should follow {model}.txt example 'checkpoint1000.txt'
		txt_pattern = re.compile(r'^' + re.escape(model_name) + r'(?i:\.' + txt_ext_pattern+ r')$')
		# Images previews should follow {model}.{extension} or {model}.preview.{extension} or {model}.{number}.{extension} or {model}.preview.{number}.{extension}
		# example 1 'checkpoint1000.png'
		# example 2 'checkpoint1000.preview.jpg'
		# example 3 'checkpoint1000.1.jpeg'
		# example 4 'checkpoint1000.preview.1.webp'
		img_pattern = re.compile(r'^' + re.escape(model_name) + r'(?i:(?:\.preview)?(?:\.\d+)?\.' + img_ext_pattern + r')$')
	elif shared.opts.model_preview_xd_name_matching == "Folder":
		# use a folder name matching that only requires the model name to show up somewhere in the folder path not the file name name
		html_pattern = html_generic_pattern
		civitai_pattern = civitai_generic_pattern
		md_pattern = md_generic_pattern
		txt_pattern = txt_generic_pattern
		prompts_pattern = prompts_generic_pattern
		img_pattern = img_generic_pattern
	else:
		# use a loose name matching that only requires the model name to show up somewhere in the file name
		html_pattern = re.compile(r'^.*' + re.escape(model_name) + r'.*(?i:\.' + html_ext_pattern + r')$')
		civitai_pattern = re.compile(r'^.*' + re.escape(model_name) + r'.*(?i:\.' + civitai_ext_pattern + r')$')
		md_pattern = re.compile(r'^.*' + re.escape(model_name) + r'.*(?i:\.' + md_ext_pattern + r')$')
		txt_pattern = re.compile(r'^.*' + re.escape(model_name) + r'.*(?i:\.' + txt_ext_pattern + r')$')
		prompts_pattern = re.compile(r'^.*' + re.escape(model_name) + r'.*(?i:\.' + prompts_ext_pattern + r')$')
		img_pattern = re.compile(r'^.*' + re.escape(model_name) + r'.*(?i:\.' + img_ext_pattern + r')$')
	
	# an array to hold the image html code
	html_code_list = []
	# if a text file is found
	found_txt_file = None
	# if a markdown file is found
	md_file = None
	# if a prompts file is found
	prompts_file = None
	# if an html file is found the iframe
	html_file_frame = None
	# if an civitai.info file is found the generated html
	civitai_info_html = None

	# if a text file is found
	generic_found_txt_file = None
	# if a markdown file is found
	generic_md_file = None
	# if a prompts file is found
	generic_prompts_file = None
	# if an html file is found the iframe
	generic_html_file_frame = None
	# if an civitai.info file is found the generated html
	generic_civitai_info_html = None

	# get the current directory so we can convert absolute paths to relative paths if we need to
	current_directory = os.getcwd()

	# support the ability to check multiple paths
	for path in paths:
		# loop through all files in the path and any subdirectories
		for dirpath, dirnames, filenames in os.walk(path, followlinks=True):
			# get a list of all parent directories
			directories = dirpath.split(os.path.sep)
			# if we are not using folder match mode look for files normally otherwise we are using folder match mode so make sure at least one parent directory is equal to the name of the model
			if shared.opts.model_preview_xd_name_matching != "Folder" or (shared.opts.model_preview_xd_name_matching == "Folder" and model_name in directories):
				# sort the file names using a natural sort algorithm
				sorted_filenames = sorted(filenames, key=natural_order_number)
				# if we are using index matching mode check to see if there is an index file, if there is read it and compile a regex that matches all the models listed in the file to use later
				index_has_model = False
				index_models_pattern = None
				if shared.opts.model_preview_xd_name_matching == "Index":
					index_txt_filename = next((filename for filename in sorted_filenames if filename.lower() == "index.txt"), None)
					if index_txt_filename is not None:
						index_txt_path = os.path.join(dirpath, index_txt_filename)
						output_text = ""
						with open(index_txt_path, "r", encoding="utf8") as file:
							output_text = file.read()
						index_models = [model.strip() for model in output_text.replace(",", "\n").splitlines()]
						index_has_model = model_name in index_models
						index_models = [re.escape(model) for model in index_models if model != model_name]
						if len(index_models) > 0:
							index_models_pattern = re.compile(r'^(?:' + r'|'.join(index_models) + r')(?i:(?:\.preview)?(?:\.\d+)?\.' + all_ext_pattern + r')$')
				# check each file to see if it is a preview file
				for filename in sorted_filenames:
					file_path = os.path.join(dirpath, filename)
					# check if the path is a subdirectory of the install directory
					is_in_a1111_dir = is_in_directory(current_directory, file_path)
					img_file = None
					# if we are using index matching, find all preview files that match the regex compiled earlier
					if shared.opts.model_preview_xd_name_matching == "Index":
						if (index_models_pattern is not None and index_models_pattern.match(filename)) or filename.lower() == "index.txt":
							# ignore preview files that strictly match any of the other models in the index file
							continue
						if index_has_model:
							if html_generic_pattern.match(filename):
								# there can only be one html file, if one was already found it is replaced
								generic_html_file_frame = create_html_iframe(file_path, is_in_a1111_dir)
							if civitai_generic_pattern.match(filename):
								# there can only be one civitai.info file, if one was already found it is replaced
								generic_civitai_info_html = create_civitai_info_html(file_path)
							if md_generic_pattern.match(filename):
								# there can only be one markdown file, if one was already found it is replaced
								generic_md_file = file_path
							if prompts_generic_pattern.match(filename):
								# there can only be one prompts file, if one was already found it is replaced
								generic_prompts_file = file_path
							if img_generic_pattern.match(filename):
								# there can be many images, even spread across the multiple paths
								img_file = file_path
							if txt_generic_pattern.match(filename):
								# there can only be one text file, if one was already found it is replaced
								generic_found_txt_file = file_path
					# perform the normal file matching rules for the matching mode determined at the beginning of this function
					if html_pattern.match(filename):
						# there can only be one html file, if one was already found it is replaced
						html_file_frame = create_html_iframe(file_path, is_in_a1111_dir)
					if civitai_pattern.match(filename):
						# there can only be one civitai.info file, if one was already found it is replaced
						civitai_info_html = create_civitai_info_html(file_path)
					if md_pattern.match(filename):
						# there can only be one markdown file, if one was already found it is replaced
						md_file = file_path
					if prompts_pattern.match(filename):
						# there can only be one prompts file, if one was already found it is replaced
						prompts_file = file_path
					if img_pattern.match(filename):
						# there can be many images, even spread across the multiple paths
						img_file = file_path
					if txt_pattern.match(filename):
						# there can only be one text file, if one was already found it is replaced
						found_txt_file = file_path
					
					# if this file was an image file append the image to the html code list
					if img_file is not None:
						html_code_list.append(create_html_img(img_file, is_in_a1111_dir))
	
	# if a generic preview file was found but not a specific one, use the generic one
	if html_file_frame is None and generic_html_file_frame is not None:
		html_file_frame = generic_html_file_frame
	if civitai_info_html is None and generic_civitai_info_html is not None:
		civitai_info_html = generic_civitai_info_html
	if md_file is None and generic_md_file is not None:
		md_file = generic_md_file
	if prompts_file is None and generic_prompts_file is not None:
		prompts_file = generic_prompts_file
	if found_txt_file is None and generic_found_txt_file is not None:
		found_txt_file = generic_found_txt_file

	# if an html file was found, ignore other txt, md, or image preview files and return the html file and prompt file if available
	if html_file_frame is not None:
		return html_file_frame, None, prompts_file, None

	# if an civitai.info file was found, ignore other txt, md, or image preview files and return the html created and prompt file if available
	if civitai_info_html is not None:
		return civitai_info_html, None, prompts_file, None

	# if there were images found, wrap the images in a container div
	html_code_output = '<div class="img-container-set">' + ''.join(html_code_list) + '</div>' if len(html_code_list) > 0 else None

	# return the all preview files found
	return html_code_output, md_file, prompts_file, found_txt_file

def get_checkpoints_dirs():
	# create list of directories

	# use the default directory as a fallback for people who want to keep their models outside of the automatic1111 directory but their preview files inside
	default_dir = os.path.join('models','Stable-diffusion') # models/Stable-diffusion
	directories = [default_dir] if os.path.exists(default_dir) and os.path.isdir(default_dir) else []

	# add the directory expected by automatic1111 for this type of model (it may be the same as the above model so only add it if its not already added)
	set_dir = shared.cmd_opts.ckpt_dir
	if set_dir is not None and os.path.exists(set_dir) and os.path.isdir(set_dir) and not is_dir_in_list(directories, set_dir):
		# WARNING: html files and markdown files that link to local files outside of the automatic1111 directory will not work correctly
		directories.append(set_dir)
	return directories

def get_embedding_dirs():
	# create list of directories

	# use the default directory as a fallback for people who want to keep their models outside of the automatic1111 directory but their preview files inside
	directories = ['embeddings', os.path.join('models','embeddings')] # support the Vladmandic fork by also adding models/embeddings as a default location
	directories = list(filter(lambda x: os.path.exists(x), directories))

	# add the directory expected by automatic1111 for this type of model (it may be the same as the above model so only add it if its not already added)
	set_dir = shared.cmd_opts.embeddings_dir
	if set_dir is not None and os.path.exists(set_dir) and os.path.isdir(set_dir) and not is_dir_in_list(directories, set_dir):
		# WARNING: html files and markdown files that link to local files outside of the automatic1111 directory will not work correctly
		directories.append(set_dir)
	return directories

def get_hypernetwork_dirs():
	# create list of directories

	# use the default directory as a fallback for people who want to keep their models outside of the automatic1111 directory but their preview files inside
	default_dir = os.path.join('models','hypernetworks') # models/hypernetworks
	directories = [default_dir] if os.path.exists(default_dir) and os.path.isdir(default_dir) else []

	# add the directory expected by automatic1111 for this type of model (it may be the same as the above model so only add it if its not already added)
	set_dir = shared.cmd_opts.hypernetwork_dir
	if set_dir is not None and os.path.exists(set_dir) and os.path.isdir(set_dir) and not is_dir_in_list(directories, set_dir):
		# WARNING: html files and markdown files that link to local files outside of the automatic1111 directory will not work correctly
		directories.append(set_dir)
	return directories

def get_lora_dirs():
	# create list of directories
	directories = []

	# add models/lora  directory as a fallback for people who want to keep their models outside of the automatic1111 directory but their preview files inside
	default_dir = os.path.join("models","Lora") # models/Lora
	if os.path.exists(default_dir) and os.path.isdir(default_dir):
		directories.append(default_dir)
	# add directories from the builtin lora extension if exists
	set_dir = shared.cmd_opts.lora_dir
	if set_dir is not None and os.path.exists(set_dir) and os.path.isdir(set_dir) and not is_dir_in_list(directories, set_dir):
		# WARNING: html files and markdown files that link to local files outside of the automatic1111 directory will not work correctly
		directories.append(set_dir)
	# add directories from the third party lora extension if exists
	if additional_networks is not None:
		# use the same pattern as the additional_networks.py extension to build up a list of paths to check for lora models and preview files
		set_dir = additional_networks.lora_models_dir
		if set_dir is not None and os.path.exists(set_dir) and os.path.isdir(set_dir) and not is_dir_in_list(directories, set_dir):
			directories.append(set_dir)
		extra_lora_path = shared.opts.data.get("additional_networks_extra_lora_path", None)
		if extra_lora_path and os.path.exists(extra_lora_path) and os.path.isdir(extra_lora_path) and not is_dir_in_list(directories, extra_lora_path):
			directories.append(extra_lora_path)
	# add models/LyCORIS if exists to support backwards compatibility with people who still have their LyCORIS models in a seperate folder
	lycoris_dir = os.path.join("models","LyCORIS") # models/LyCORIS
	if os.path.exists(lycoris_dir) and os.path.isdir(lycoris_dir):
		directories.append(lycoris_dir)
	return directories

def get_lycoris_dirs():
	# create list of directories
	directories = []

	# add directories from the third party lycoris extension if exists
	if lycoris_module is not None:
		# add models/LyCORIS just in case to the list of directories
		default_dir = os.path.join("models","LyCORIS") # models/LyCORIS
		if os.path.exists(default_dir) and os.path.isdir(default_dir):
			directories.append(default_dir)
		# add directories from the third party lycoris extension if exists
		set_dir = shared.cmd_opts.lyco_dir
		if set_dir is not None and os.path.exists(set_dir) and os.path.isdir(set_dir) and not is_dir_in_list(directories, set_dir):
			# WARNING: html files and markdown files that link to local files outside of the automatic1111 directory will not work correctly
			directories.append(set_dir)

	return directories

def show_model_preview(modelname=None):
	# get preview for the model
	return show_preview(modelname, get_checkpoints_dirs(), "checkpoints")

def show_embedding_preview(modelname=None):
	# get preview for the model
	return show_preview(modelname, get_embedding_dirs(), "embeddings")

def show_hypernetwork_preview(modelname=None):
	# get preview for the model
	return show_preview(modelname, get_hypernetwork_dirs(), "hypernetworks")

def show_lora_preview(modelname=None):
	# get preview for the model
	return show_preview(modelname, get_lora_dirs(), "loras")

def show_lycoris_preview(modelname=None):
	# get preview for a LyCORIS
	return show_preview(modelname, get_lycoris_dirs(), "lycoris")

def show_preview(modelname, paths, tags_key):
	if modelname is None or len(modelname) == 0 or paths is None or len(paths) == 0:
		txt_update = gr.Textbox.update(value=None, visible=False)
		md_update = gr.Textbox.update(value=None, visible=False)
		prompts_list_update = gr.CheckboxGroup.update(visible=False)
		prompts_button_update = gr.Button.update(visible=False)
		html_update = gr.HTML.update(value='', visible=False)
		tags_html = gr.HTML.update(value='', visible=False)
		return prompts_list_update, prompts_button_update, txt_update, md_update, html_update, tags_html
	
	# remove the hash if exists, the extension, and if the string is a path just return the file name
	name = clean_modelname(modelname)
	# get the preview data
	html_code, found_md_file, found_prompts_file, found_txt_file = search_and_display_previews(name, paths)
	preview_html = '' if html_code is None else html_code

	# if a text file was found update the gradio text element
	if found_txt_file:
		output_text = ""
		with open(found_txt_file, "r", encoding="utf8") as file:
			for line in file:
				output_text = f'{output_text}{line.strip()}\n'
		txt_update = gr.Textbox.update(value=output_text, visible=True)
	else:
		txt_update = gr.Textbox.update(value=None, visible=False)
	
	# if a markdown file was found update the gradio markdown element
	if found_md_file:
		output_text = ""
		with open(found_md_file, "r", encoding="utf8") as file:
			output_text = file.read()
		md_update = gr.Textbox.update(value=output_text, visible=True)
	else:
		md_update = gr.Textbox.update(value=None, visible=False)

	# if a prompt file was found update the gradio prompts list
	if found_prompts_file:
		prompts: list[str] = list()
		try:
			with open(found_prompts_file, newline='') as csvfile:
				reader = csv.reader(csvfile)
				for row in reader:
					for prompt in row:
						if prompt not in prompts:
							prompts.append(prompt)
		finally:
			prompts_list_update = gr.CheckboxGroup.update(visible=True, choices=prompts, value=prompts)
			prompts_button_update = gr.Button.update(visible=True)
	else:
		prompts_list_update = gr.CheckboxGroup.update(visible=False)
		prompts_button_update = gr.Button.update(visible=False)

	# if images were found or an HTML file was found update the gradio html element
	if html_code:
		html_update = gr.HTML.update(value=preview_html, visible=True)
	else:
		html_update = gr.HTML.update(value='', visible=False)

	# if nothing was found display a message that nothing was found
	if found_txt_file is None and found_md_file is None and (html_code is None or len(html_code) == 0):
		html_update = gr.HTML.update(value="<span style='margin-left: 1em;'>No Preview Found</span>", visible=True)

	# get the tags from the tags object and create a span for them
	found_tags = tags[tags_key].get(modelname, None)
	if found_tags is not None:
		tags_html = gr.HTML.update(value=f'<div class="footer-tags">{found_tags}</div>', visible=True)
	else:
		tags_html = gr.HTML.update(value='', visible=False)
	return prompts_list_update, prompts_button_update, txt_update, md_update, html_update, tags_html

def create_tab(tab_label, tab_id_key, list_choices, show_preview_fn, filter_fn, refresh_fn, update_selected_fn):
	# create a tab for model previews
	with gr.Tab(tab_label, elem_id=f"model_preview_xd_{tab_label.lower()}_tab", elem_classes="model_preview_xd_tab"):
		with gr.Row(elem_id=f"{tab_id_key}_modelpreview_xd_control_row", elem_classes="modelpreview_xd_control_row"):
			list = gr.Dropdown(label="Model", choices=list_choices, interactive=True, elem_id=f"{tab_id_key}_mp2_preview_model_list", elem_classes="mp2_preview_model_list")
			filter_input = gr.Textbox(label="Filter", value="", elem_id=f"{tab_id_key}_modelpreview_xd_filter_text", elem_classes="modelpreview_xd_filter_text")
		with gr.Row(elem_id=f"{tab_id_key}_modelpreview_xd_hidden_ui", elem_classes="modelpreview_xd_hidden_ui"):
			refresh_list = gr.Button(value=refresh_symbol, elem_id=f"{tab_id_key}_modelpreview_xd_refresh_sd_model", elem_classes="modelpreview_xd_refresh_sd_model")
			update_model_input = gr.Textbox(value="", elem_id=f"{tab_id_key}_modelpreview_xd_update_sd_model_text", elem_classes="modelpreview_xd_update_sd_model_text")
			update_model_button = gr.Button(value=update_symbol, elem_id=f"{tab_id_key}_modelpreview_xd_update_sd_model", elem_classes="modelpreview_xd_update_sd_model")
		with gr.Row(elem_id=f"{tab_id_key}_modelpreview_xd_prompts_row", elem_classes="modelpreview_xd_prompts_row"):
			prompts_list = gr.Checkboxgroup(label="Prompts", visible=False, interactive=True, elem_id=f"{tab_id_key}_modelpreview_xd_prompts_list", elem_classes="modelpreview_xd_prompts_list")
			prompts_copy_button = gr.Button(value="Copy", visible=False, interactive=True, elem_id=f"{tab_id_key}_modelpreview_xd_prompts_copy_button", elem_classes="modelpreview_xd_prompts_copy_button")
		with gr.Row(elem_id=f"{tab_id_key}_modelpreview_xd_notes_row", elem_classes="modelpreview_xd_notes_row"):
			notes_text_area = gr.Textbox(label='Notes', interactive=False, lines=1, visible=False, elem_id=f"{tab_id_key}_modelpreview_xd_update_sd_model_text_area", elem_classes="modelpreview_xd_update_sd_model_text_area")
		with gr.Row(elem_id=f"{tab_id_key}_modelpreview_xd_html_row", elem_classes="modelpreview_xd_html_row"):
			with gr.Row(elem_id=f"{tab_id_key}_modelpreview_xd_flexcolumn_row", elem_classes="modelpreview_xd_flexcolumn_row"):
				preview_html = gr.HTML(elem_id=f"{tab_id_key}_modelpreview_xd_html_div", elem_classes="modelpreview_xd_html_div", visible=False)
				preview_md = gr.Markdown(elem_id=f"{tab_id_key}_modelpreview_xd_markdown_div", elem_classes="modelpreview_xd_markdown_div", visible=False)
		with gr.Row(elem_id=f"{tab_id_key}_modelpreview_xd_tags_row", elem_classes="modelpreview_xd_tags_row"):
			preview_tags = gr.HTML(elem_id=f"{tab_id_key}_modelpreview_xd_tags_div", elem_classes="modelpreview_xd_tags_div", visible=False)

	list.change(
		fn=show_preview_fn,
		inputs=[
			list,
		],
		outputs=[
			prompts_list,
			prompts_copy_button,
			notes_text_area,
			preview_md,
			preview_html,
			preview_tags
		]
	)

	filter_input.change(
		fn=filter_fn,
		inputs=[
			filter_input,
		],
		outputs=[
			list,
		]
	)

	refresh_list.click(
		fn=refresh_fn,
		inputs=[
			list,
			filter_input,
		],
		outputs=[
			list,
			prompts_list,
			prompts_copy_button,
			notes_text_area,
			preview_md,
			preview_html,
			preview_tags
		]
	)

	update_model_button.click(
		fn=update_selected_fn,
		inputs=[
			update_model_input,
		],
		outputs=[
			list,
			prompts_list,
			prompts_copy_button,
			notes_text_area,
			preview_md,
			preview_html,
			preview_tags
		]
	)

	prompts_copy_button.click(
		fn=None,
		inputs=[prompts_list],
		_js="(x) => copyToClipboard(x)",
	)

def on_ui_tabs():
	global additional_networks, additional_networks_builtin
	# import/update the lora module
	additional_networks = import_lora_module()
	additional_networks_builtin = import_lora_module_builtin()
	lycoris_module = import_lycoris_module()

	# create a gradio block
	with gr.Blocks() as modelpreview_interface:

		limitHeight = shared.opts.model_preview_xd_limit_sizing
		columnView = shared.opts.model_preview_xd_column_view

		gr.HTML(elem_id='modelpreview_xd_setting', value='<script id="modelpreview_xd_setting_json" type="application/json">{ "LimitSize": ' + ( "true" if limitHeight else "false" ) + ', "ColumnView": ' + ( "true" if columnView else "false" ) + ' }</script>', visible=False)

		# create a tab for the checkpoint previews
		create_tab("Checkpoints", "cp",
				list_all_models(),
				show_model_preview,
				filter_models,
				refresh_models,
				update_checkpoint)
		create_tab("Embeddings", "em",
				list_all_embeddings(),
				show_embedding_preview,
				filter_embeddings,
				refresh_embeddings,
				update_embedding)
		create_tab("Hypernetwork", "hn",
				list_all_hypernetworks(),
				show_hypernetwork_preview,
				filter_hypernetworks,
				refresh_hypernetworks,
				update_hypernetwork)

		# create a tab for the lora previews if the module was loaded
		if additional_networks is not None or additional_networks_builtin is not None:
			create_tab("Lora", "lo",
						list_all_loras(),
						show_lora_preview,
						filter_loras,
						refresh_loras,
						update_lora)

		# create a tab for the LyCORIS previews if the module was loaded
		if lycoris_module is not None:
			create_tab("LyCORIS", "ly",
					   list_all_lycorii(),
					   show_lycoris_preview,
					   filter_lycorii,
					   refresh_lycorii,
					   update_lycorii)
	
	return (modelpreview_interface, "Model​ Previews", "modelpreview_xd_interface"),

def on_ui_settings():
	section = ('model_preview_xd', "Model Preview XD")
	shared.opts.add_option("model_preview_xd_name_matching", shared.OptionInfo("Loose", "Name matching rule for preview files", gr.Radio, {"choices": ["Loose", "Strict", "Folder", "Index"]}, section=section).info("Requires UI Reload").html("""
<ul style='margin-left: 1.5em'>
	<li><strong>Loose</strong> - Use a loose naming scheme for matching preview files. Your preview files must contain the model name somewhere in their file name. If your model is named <strong>'model.ckpt'</strong> your preview files must be named in the following manner:
		<ul style='margin-left: 2em'>
			<li>my<strong>model</strong>.html</li>
			<li><strong>model</strong>_markdown.md</li>
			<li>trigger_words_for_<strong>model</strong>.txt</li>
			<li><strong>model</strong>-image.webp</li>
			<li><strong>model</strong>.preview.png</li>
			<li>my3D<strong>model</strong>.jpg</li>
			<li><strong>model</strong>ling.jpeg</li>
		</ul>
	</li>
	<li><strong>Strict</strong> - Use a strict naming scheme for matching preview files. If your model is named <strong>'model.ckpt'</strong> your preview files must be named in the following manner:
		<ul style='margin-left: 2em'>
			<li><strong>model</strong>.html</li>
			<li><strong>model</strong>.md</li>
			<li><strong>model</strong>.txt</li>
			<li><strong>model</strong>.webp</li>
			<li><strong>model</strong>.preview.png</li>
			<li><strong>model</strong>.3.jpg</li>
			<li><strong>model</strong>.preview.4.jpeg</li>
		</ul>
	</li>
	<li><strong>Folder</strong> - Use folder name matching. Will look for a folder within your model directory that matches your model's name (case sensitive) and will show any preview files found within that folder or any subfolders of that folder. If your model is named <strong>'mymodel.ckpt'</strong> all preview files located in <strong>'/mymodel/'</strong> will be shown.</li>
	<li><strong>Index</strong> - If a folder contains a file <strong>'index.txt'</strong> that lists model names, any preview files in that folder regardless of name will be associated with each model in the index file. This allows you to share preview files among a number of models. This matching mode will also match any file named similar to the <strong>'Strict'</strong> matching mode to allow you to still specify preview files for specific models.</li>
</ul>"""))
	shared.opts.add_option("model_preview_xd_limit_sizing", shared.OptionInfo(True, "Limit the height of previews to the height of the browser window", section=section).info(".html preview files are always limited regardless of this setting. Requires UI Reload"))
	shared.opts.add_option("model_preview_xd_column_view", shared.OptionInfo(False, "Column view", section=section).info("This is only recommended if you use .txt files. Left column will have model select, .txt and .prompt preview data. Right column will have preview images and .md preview data, or .civitai.info preview data or .html preview data. Requires UI Reload"))
	shared.opts.add_option("model_preview_xd_cache_images_civitai_info", shared.OptionInfo(False, "Cache images from .civitai.info previews", section=section).info("Saves files to extension folder."))

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)