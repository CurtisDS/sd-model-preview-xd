# sd-model-preview-xd
Extension for [Automatic1111 Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) to display previews for custom models[^1].

[^1]: This extension should support all operating systems in theory but has only been tested in Windows

## About
With so many new models appearing it's becoming harder to keep track of what models output what styles and what tags are used to call these styles.
This extension allows you to create various types of preview files/notes[^preview-files], each with the same name as your model and have this info easily displayed for later reference in webui.

[^preview-files]: Generally referred to as "preview files" this refers to any file type supported by this extension including `.html`, `.txt`, `.png` etc.

## Installation
1. From the extensions tab in web ui click install from url
2. Paste `https://github.com/CurtisDS/sd-model-preview-xd` into the url box.
3. Click Install
4. From the Installed tab click apply and restart[^2].
5. Put `.html`[^3], `.md`[^3], `.txt`, `.png`, `.webp`, `.jpg`/`.jpeg`, `.prompt`[^4], `.tags`[^5], and/or `.civitai.info`[^6]  files in the same directory as your models, or a subfolder. Make sure to name the files the same as your model. You may append something after the name of the model and it will still work ([See: Name Matching Rules](#name-matching-rules)). You can have multiple images for a single model, but only one markdown, text, civitai.info or html file. You can also mix and match any of the preview files except for HTML and civitai.info files, if the extension finds either file it will only show that output (html files have priority).

[^2]: If you run into issues after first install you may need to fully shutdown and rerun the webui-user.bat after you install the extension.
[^3]: HTML and Markdown files will not support linking to files or images outside of the Automatic1111 directory. If you cannot keep linked files within the install directory upload them to the internet and link to them remotely.
[^4]: A `.prompt` file is a CSV file containing a list of prompts associated with your model.
[^5]: A `.tags` file is just a text file containing words that you want to use for searching for the associated model. It is suggested you format this as a list of hashtags. For example: `"#anime #sfw #mix #high_quality"`. This will help avoid matching `"nsfw"` when searching for `"sfw"`. However there is no *required* format. A search will match as long as the search text appears anywhere in the file.
[^6]: A `.civitai.info` file is a JSON file created by the [Civitai Helper extension](https://github.com/butaixianran/Stable-Diffusion-Webui-Civitai-Helper)


***Note**: If you are using symlinks or otherwise changing the default model directories [click here for information](#changing-default-directories)*

For example if you have `my_model.ckpt` in `models\Stable-diffusion` then you can put `my_model.txt`, `my_model_1.jpg` and, `my_model_2.jpg` in `models\Stable-diffusion\Previews` and it will display them in the "Model Preview" tab.

This extension supports the following model types in the the default directories:

- SD Checkpoints
- Embeddings
- Hypernetworks
- LoRA

## Usage
1. After creating the preview files and putting them in the corresponding directories, select the Model Preview tab in web ui and then the type of model you want to preview
2. Select a model from the dropdown list. (If the model has any preview files they will be shown)
3. Any preview png files found that also contain prompt data embedded in them will have a red "copy" button when hovering over the image. By clicking the button it will copy the prompt data to your clipboard.
4. If you would like to filter the list of models enter text in the filter text box. The filter text will be separated by commas and return models who have that text anywhere in its name or its associated `.tags`[^5] file.

![screenshot](https://github.com/CurtisDS/sd-model-preview-xd/raw/main/sd-model-preview-xd.png)

## Tips

1. <a name="tips-1"></a>You can save the `README.md` file from a models huggingface page to use as your model preview:

      ![screenshot of markdown example](https://github.com/CurtisDS/sd-model-preview-xd/raw/main/sd-model-preview-xd-markdown-example.png)

2. <a name="tips-2"></a>You can save the model's Civitai page (using <kbd>Ctrl</kbd>+<kbd>S</kbd> in your browser) to use the Civitai page as your model preview:
      
      ![screenshot of html example](https://github.com/CurtisDS/sd-model-preview-xd/raw/main/sd-model-preview-xd-html-example.png)

3. <a name="tips-3"></a>If you want to keep it clean, create a simple .txt file with the trigger words and save a few sample images:

      ![screenshot of text and images example](https://github.com/CurtisDS/sd-model-preview-xd/raw/main/sd-model-preview-xd-text-and-image-example.png)

4. <a name="tips-4"></a>You can now be linked directly to the preview files of a model by clicking on the `ⓘ` in the extra networks thumbnail cards.

      - When `Default view for Extra Networks` is set to `cards`:

          ![screenshot of extra networks card link to preview](https://user-images.githubusercontent.com/20732674/216813267-c8539ae3-c318-42fa-b5db-f89176993fbc.png)

      - When `Default view for Extra Networks` is set to `thumbs`:

          ![screenshot of extra networks card link to preview](https://user-images.githubusercontent.com/20732674/216813283-2e4f874f-3afa-4088-98eb-95bff0566ec8.png)

5. <a name="tips-5"></a>In the settings for the extension you can turn off the `limit height` setting to change how tall the preview panel can be.
      
      ***Note**: An update to gradio has caused an issue that cuts off the model select dropdown when using the `limit height` option so it has been removed for now. If I can get it to work again I will bring it back.*

      ![screenshot of difference between limit height and not limit height](https://github.com/CurtisDS/sd-model-preview-xd/raw/main/sd-model-preview-xd-height-limit.gif)

## Things to watch out for

### Name Matching Rules

In the settings tab there is a page for Model Preview XD where you can update the setting to use "Strict", "Loose", "Folder", or "Index" naming. Depending on that setting the rules for naming preview files will be slightly different.

- #### Strict Name Matching:

  Name your preview files the exact same as the model name. To support multiple images you can also choose to append to the model name `.preview` and/or `.1` (where 1 can be any number). 

  Here are a number of examples that will work with strict naming assuming your model is named `model.ckpt`:

  - model.txt
  - model.md
  - model.html
  - model.png
  - model.preview.png
  - model.4.png
  - model.preview.7.png
  - model.tags

  ***Note** that in the example png images were used but you can use png, jpg, jpeg, or webm images*

- #### Loose Name Matching:

  The naming rule for loose name matching is that your model name has to appear anywhere in the file name. Please note this has the potential to return preview files for other models that are named similarly. For example, if you have a model named `my-checkpoint.ckpt` and `my-checkpoint2.ckpt` the extension will pick up preview files meant for `my-checkpoint2` in its search for preview files for `my-checkpoint`. You can avoid this my renaming `my-checkpoint` to `my-checkpoint1` (*Make sure to also update any existing preview files*).

  Here are a number of examples that will work with loose naming assuming your model is named `model.ckpt`:

  - model_trigger_words.txt
  - model_readme.md
  - my_model_webpage.html
  - model.png
  - model_preview.png
  - model.image.png
  - 3D_modelling_checkpoint.png
  - model_tags.tags

  ***Note** that in the example png images were used but you can use png, jpg, jpeg, or webm images*

  ***Also note** that preview files that appear to be for other checkpoints have also been returned*

- #### <u>Folder Name Matching:</u>

  When using folder name matching the extension will look for a folder matching your model name and return any preview files found within, including subdirectories.

  Here are a number of examples that will work with folder naming assuming your model is named `model.ckpt`:

  - /model/trigger_words.txt
  - /model/readme.md
  - /model/info.html
  - /model/0.png
  - /model/preview.png
  - /model/filters.tags

  ***Note** that in the example png images were used but you can use png, jpg, jpeg, or webm images*

- #### <u>Index Matching:</u>

  This acts like [Strict Name Matching](#strict-name-matching) with an addition feature. On top of matching preview files in the same manner as Strict Name Matching this mode is intended to also optionally allow multiple models to share preview files in certain cases. To do this, create a file `index.txt` and place it in a folder that contains preview files you want to be shared across models. Index files should contain the model's file names (not including extensions) each separated by a new line.
  
  For example:
  ```text
  MyModel2000_Steps
  MyModel3000_Steps
  MyModel4000_Steps
  ```
  
  When the extension finds an index file it will scan it for model names and then scan the rest of that folder for any preview files (regardless of naming convention) and apply all preview files it finds to each model that is listed in the index file. For any model you have listed in the index file if it finds a preview file that matches the [Strict Name Matching](#strict-name-matching) convention it will only apply that preview file to that specific model. Also for preview files where you are limited to one of that type of preview file (Such as `.txt`, `.md`, and `.html` files) it will match the strict version if one exists over the general one.

  For example if you have the following files in a folder:

  - /MyModel/index.txt
  - /MyModel/common.png
  - /MyModel/trigger_words.txt
  - /MyModel/MyModel2000_Steps.png
  - /MyModel/MyModel2000_Steps.txt
  - /MyModel/MyModel3000_Steps.png

  `MyModel2000_Steps` will match: `common.png`, `MyModel2000_Steps.png`, and `MyModel2000_Steps.txt`.<br>
  `MyModel3000_Steps` will match: `common.png`, `MyModel3000_Steps.png`, and `trigger_words.txt`.<br>
  `MyModel4000_Steps` will match: `common.png`, and `trigger_words.txt`.

  ***Note** that using an index file is optional and the default behavior is otherwise identical to the Strict Name Matching mode*
  
  ***Also note** that shared preview files can be named anything you want as long as they have the appropriate file extensions. The file names `common.png` and `trigger_words.txt` were only used as examples.*


### Changing Default Directories

Gradio (the software Automatic1111 is built on) doesn't support linking to files outside of the Automatic1111 install directory through the webui. So if you have used symlinks or Automatic1111's built in command line arguments to change the directories for your models to something outside of the Automatic1111 directory you will need to take advantage of one of the following workarounds for your previews to work.

1. If you use the command line arguments to change the directories for your models the extension will look in both the default directories and the custom directories for preview files. So you could change the directory for your models and leave your preview files in the default directories, this will keep them within the install directory and remove the issues with linking.

      <details>
      <summary>Click here for a quick guide on how to change directories without using symlinks.</summary>
      ​

      If you want to change the directories for models add these settings to your `webui-user.bat` `COMMANDLINE_ARGS` for each model type:

      `--ckpt-dir "D:\\my models\\checkpoints"`

      `--hypernetwork-dir "D:\\my models\\hypernetworks"`

      `--embeddings-dir "D:\\my models\\embeddings"`

      `--lora-dir "D:\\my models\\lora"`

      If you wanted to use all the settings at once your COMMANDLINE_ARGS line would look something like this:

      ```bash
      set COMMANDLINE_ARGS=--xformers --api --ckpt-dir "D:\\my models\\checkpoints" --hypernetwork-dir "D:\\my models\\hypernetworks" --embeddings-dir "D:\\my models\\embeddings" --lora-dir "D:\\my models\\lora"
      ```

      </details>

2. The extension can detect if a preview file is outside of the install directory and alter how it handles the preview to try and avoid some of the issues with linking files in the webui. The following differences will occur:
- **Text files**: Nothing will change, it will work the same as if it was in the install directory.

- **Image files**: The images will be converted to a base64 string essentially copying the image into the html code instead of linking to a file. This may slightly increase load times but should be otherwise fine.

- **Markdown files**: The preview will load but if you linked to a local image or file in the markdown - even if that file or image is in the same directory as the markdown file - it may not resolve that link. A workaround would be to upload files or images to the internet and link to them remotely instead of locally, then the links will resolve.

- **HTML files**: Normally the extension will create an `<iframe>` linking to the HTML file, however again because you cannot link to files it will now convert the file to a base64 string and use that in the `<iframe>` instead. Also, as with markdown files if you linked to a local image or file in the HTML code it may not resolve that link. You can use the same workaround, though, which is to upload the files or images to the internet and link to them remotely instead of locally.

### Linking to local files/images in markdown or html pages

Linking to a file/image using relative paths is slightly different in Markdown vs HTML because of the difference in how they are loaded. Markdown has the relative path resolve from the location of the Automatic1111 install directory where as HTML files will need to have the path be relative from the actual HTML file.... unless you have the HTML outside of the Automatic1111 directory which changes how the HTML file is loaded and also will change where the path is relative from.

A way around this is to link to the file using the webserver that is created by Automatic1111. By default it is located at `http://127.0.0.1:7860`. This webserver serves up all files within the install directory and so will have relative paths be relative to that directory. The webserver doesn't have access to anything outside of the install directory so you will get an error if you try linking to any such file.

You will also have to use a special URI to reference that you want to use a local file. You need to include `file=` before the file path.

So in conclusion if you would like to link to a file or image from within a Markdown or HTML file use this syntax with the path always being relative from the Automatic install directory:

```html
<img src="http://127.0.0.1:7860/file=models/Stable-diffusion/image.png">
```

```markdown
![image alt text](http://127.0.0.1:7860/file=models/Stable-diffusion/image.png)
```
