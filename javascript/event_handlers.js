let previewTab = null;

// Sync the refresh button for list with the invisible refresh button in the extension
function registerClickEvents(refreshButton, invisible_button_id_selectors) {
  // Check if the button element exists and is not null
  if (typeof refreshButton != "undefined" && refreshButton != null) {
    // Only register a new event if you haven't already
    if (refreshButton.getAttribute('md_preview_listener') !== 'true') {
      // Set this attribute to true so we don't set the same listener again
      refreshButton.setAttribute('md_preview_listener', 'true');
      // Add a click event listener to the main refresh model button
      refreshButton.addEventListener('click', (event) => {
        // For each button selector in the button selector list, click that button
        invisible_button_id_selectors.forEach(invisible_button_id_selector => {
          // Get the preview refresh model button element
          const previewRefreshModelButton = gradioApp().querySelector(invisible_button_id_selector);
          // Check if the button element exists and is not null
          if (typeof previewRefreshModelButton != "undefined" && previewRefreshModelButton != null) {
            // Click the preview refresh model button after 500ms (gives it a bit to load)
            setTimeout((event) => {
              previewRefreshModelButton.click();
            }, 500);
          }
        });
      });
    }
  }
}

// Sync the value of the selected model from the main SD model list to the model list of this extension
function setSelectValue(selectPreviewModelElement, selectSDModelElement) {
  // Check if the element exists, is not null, and its value is different than the SD model checkpoint
  if(typeof selectPreviewModelElement != "undefined" && selectPreviewModelElement != null &&
    selectPreviewModelElement.value != selectSDModelElement.value) {
    // Set the value of the preview model list to the value of the SD model checkpoint
    selectPreviewModelElement.value = selectSDModelElement.value;
    // Get the options of the preview model list as an array
    const options = Array.from(selectPreviewModelElement.options);
    // Find the option in the array that has the same text as the SD model checkpoint value
    const optionToSelect = options.find(item => item.text == selectSDModelElement.value);
    // Check if the option was found and is not null
    if(typeof optionToSelect != "undefined" && optionToSelect != null) {
      // Mark the option as selected
      optionToSelect.selected = true;
      // Dispatch a change event on the preview model list
      selectPreviewModelElement.dispatchEvent(new Event('change'));
    }
  }
}

// Is fired by automatic1111 when the UI is updated
onUiUpdate(function() {
  // There is a script containing settings for height
  // Get the element and set the attribute to limit the height
  let tabEl = gradioApp().getElementById('tab_modelpreview_xd_interface');
  let jsonEl = gradioApp().getElementById('modelpreview_xd_setting_json');
  if (typeof tabEl != 'undefined' && tabEl != null && typeof jsonEl != 'undefined' && jsonEl != null) {
    let settingsJSON = JSON.parse(jsonEl.innerHTML);
    if (settingsJSON.LimitSize) {
      tabEl.setAttribute('limit-height', '');
    }
    if (settingsJSON.ColumnView) {
      tabEl.setAttribute('column-view', '');
    }
  }

  // Get the select element for the SD model checkpoint
  const selectSDModelElement = gradioApp().querySelector('#setting_sd_model_checkpoint select');
  // Check if the element exists and is not null
  if(typeof selectSDModelElement != "undefined" && selectSDModelElement != null) {
    // Get the select element for the preview model list
    const selectPreviewModelElement = gradioApp().querySelector('#cp_mp2_preview_model_list select');
    // Only register a new event if you haven't already
    if (selectPreviewModelElement != "undefined" && selectPreviewModelElement != null && selectSDModelElement.getAttribute('md_preview_listener') !== 'true') {
      // Set this attribute to true so we don't set the same listener again
      selectSDModelElement.setAttribute('md_preview_listener', 'true');
      // Add an event handler that will update the select with the new value if someone changes the checkpoint
      selectSDModelElement.addEventListener('change', (event) => setSelectValue(selectPreviewModelElement, selectSDModelElement));
      // Check if the element exists, is not null, but its value is not set
      if(typeof selectPreviewModelElement != "undefined" && selectPreviewModelElement != null &&
        (typeof selectPreviewModelElement.value == "undefined" || selectPreviewModelElement.value == null ||
        selectPreviewModelElement.value == "")) {
        // Set the value after 100ms
        setTimeout((event) => {
          setSelectValue(selectPreviewModelElement, selectSDModelElement);
        }, 500);
      }
    }
  }

  // Sync the refresh main model list button
  registerClickEvents(gradioApp().querySelector('#refresh_sd_model_checkpoint'), ['#cp_modelpreview_xd_refresh_sd_model','#lo_modelpreview_xd_refresh_sd_model','#ly_modelpreview_xd_refresh_sd_model','#hn_modelpreview_xd_refresh_sd_model','#em_modelpreview_xd_refresh_sd_model']);

  // Sync the new refresh extra network buttons to this extension
  registerClickEvents(gradioApp().querySelector('#txt2img_extra_refresh'), ['#cp_modelpreview_xd_refresh_sd_model','#lo_modelpreview_xd_refresh_sd_model','#ly_modelpreview_xd_refresh_sd_model','#hn_modelpreview_xd_refresh_sd_model','#em_modelpreview_xd_refresh_sd_model']);
  registerClickEvents(gradioApp().querySelector('#img2img_extra_refresh'), ['#cp_modelpreview_xd_refresh_sd_model','#lo_modelpreview_xd_refresh_sd_model','#ly_modelpreview_xd_refresh_sd_model','#hn_modelpreview_xd_refresh_sd_model','#em_modelpreview_xd_refresh_sd_model']);

  // get the main tabs and specifically the button to switch to the preview tab
  let tabs = gradioApp().querySelectorAll("#tabs > div:first-of-type button");
  if(typeof tabs != "undefined" && tabs != null && tabs.length > 0) {
    tabs.forEach(tab => {
      if(tab.innerText == "Model​ Previews") {
        previewTab = tab;
      }
    });

    if(previewTab == null && typeof tabEl != 'undefined' && tabEl != null) {
      // the tab button wasn't found so use this backup method to find the proper tab button
      // get the parent of the extensions tab and find the position within its parent. This will be the tabs div.
      // note that the parent contains a div for the tab button row and a div for each tab content
      // so the button position will be 1 less than the index
      var tabIndex = Array.from(tabEl.parentElement.children).indexOf(tabEl);
      if(tabs.length >= tabIndex) {
        previewTab = tabs[tabIndex - 1];
      }
    }
  }

  // get the thumb cards and inject a link that will pop the user back to the preview tab for that model
  let thumbCards = gradioApp().querySelectorAll("#txt2img_extra_tabs .card:not([preview-hijack]), #img2img_extra_tabs .card:not([preview-hijack])");
  if(typeof thumbCards != "undefined" && thumbCards != null && thumbCards.length > 0) {
    thumbCards.forEach(card => {
      let buttonRow = card.querySelector('.button-row');
      // the name of the model is stored in a span beside the .additional div
      let modelName = card.getAttribute('data-name');
      
      // get the id of the parent div to check what type of model this thumb card is for
      let cardNetwork = card.parentNode;
      let id = cardNetwork.getAttribute("id");

      let modelToSelect = '';
      
      // get the correct tab string for this type of model
      switch(id) {
        case "txt2img_textual inversion_cards":
        case "img2img_textual inversion_cards":
        case "txt2img_textual_inversion_cards":
        case "img2img_textual_inversion_cards":
          modelToSelect = "Embeddings";
        break;
        case "txt2img_hypernetworks_cards":
        case "img2img_hypernetworks_cards":
          modelToSelect = "Hypernetwork";
        break;
        case "txt2img_lora_cards":
        case "img2img_lora_cards":
          modelToSelect = "Lora";
        break;
        case "txt2img_lycoris_cards":
        case "img2img_lycoris_cards":
          modelToSelect = "LyCORIS";
        break;
        case "txt2img_checkpoints_cards":
        case "img2img_checkpoints_cards":
          modelToSelect = "Checkpoints";
        break;
      }

      // build out a new button to link to the preview tab
      let previewXD_Btn = document.createElement("div");
      previewXD_Btn.className = "previewXD-button card-button info";
      previewXD_Btn.title = "Go To Preview";
      previewXD_Btn.onclick = function(event) {
          doCardClick(event, modelName, modelToSelect);
      };
      buttonRow.prepend(previewXD_Btn);

      // we are finished so add the hijack attribute so we know not we don't need to do this card again
      card.setAttribute("preview-hijack", true);
    });
  }

  //##################### Hide Subdir Buttons That Start With '_' ##########################

   // Find all divs with class "extra-network-subdirs"
   var subdirDivs = document.querySelectorAll(".extra-network-subdirs");

   // Iterate through each div
   subdirDivs.forEach(function(div) {
       // Find all buttons within the current div
       var buttons = div.querySelectorAll("button");

       // Iterate through each button
       buttons.forEach(function(button) {
           // Check if the button text starts with "_"
           if (button.textContent.trim().startsWith("_")) {
               // Hide the button
               button.style.display = "none";
           }
       });
   });

   //########################################################################################
})

function doCardClick(event, name, modelType) {
  // prevent the <a> tag from linking anywhere and prevent the default click action for clicking on a card
  event.stopPropagation();
  event.preventDefault();

  // get the tabs that are in the preview extension
  let tabs = gradioApp().querySelectorAll("#tab_modelpreview_xd_interface > div:first-of-type button");
  if(typeof tabs != "undefined" && tabs != null && tabs.length > 0) {
    foundTab = null;
    tabs.forEach(tab => {
      if(tab.innerText.trim() == modelType) {
        foundTab = tab;
      }
    });

    if(foundTab == null) {
      // use backup method for finding the button if string compare doesn't work. This is a temporary fix until new gradio update with button ids.
      switch (modelType) {
        case "Hypernetwork":
          foundTab = tabs[2];
        break;
        case "Lora":
          foundTab = tabs[3];
        break;
        case "Embeddings":
          foundTab = tabs[1];
        break;
        case "Checkpoints":
          foundTab = tabs[0];
        break;
      }
    }

    if(foundTab != null) {
      tab = foundTab;

      // click on the tab to activate it
      tab.click();
      
      let modelNameID = null;
      let modelUpdateID = null;

      // get the appropriate ids for the invisible text and button elements that will let us programmatically set the dropdown later
      switch (modelType) {
        case "Hypernetwork":
          modelNameID = "hn_modelpreview_xd_update_sd_model_text";
          modelUpdateID = "hn_modelpreview_xd_update_sd_model";
        break;
        case "Lora":
          modelNameID = "lo_modelpreview_xd_update_sd_model_text";
          modelUpdateID = "lo_modelpreview_xd_update_sd_model";
        break;
        case "LyCORIS":
          modelNameID = "ly_modelpreview_xd_update_sd_model_text";
          modelUpdateID = "ly_modelpreview_xd_update_sd_model";
        break;
        case "Embeddings":
          modelNameID = "em_modelpreview_xd_update_sd_model_text";
          modelUpdateID = "em_modelpreview_xd_update_sd_model";
        break;
        case "Checkpoints":
          modelNameID = "cp_modelpreview_xd_update_sd_model_text";
          modelUpdateID = "cp_modelpreview_xd_update_sd_model";
        break;
      }

      // get the text area and the button
      let modelName = gradioApp().querySelector(`#${modelNameID} textarea`);
      let modelUpdate = gradioApp().querySelector(`#${modelUpdateID}`);

      if(typeof modelName != "undefined" && modelName != null && 
          typeof modelUpdate != "undefined" && modelUpdate != null) {
        // set the textarea's value
        modelName.value = name;
        
        // dispatch an event to trigger the gradio update for the textarea
        const inputEvent = new Event("input");
        modelName.dispatchEvent(inputEvent);
        // click the update button to trigger the python code to set the dropdown
        modelUpdate.click();
        // click on the model preview tab now that we have selected the right preview
        setTimeout((event) => {
          previewTab.click();
        }, 100);
      }
    }
  }
}

function metaDataCopy(event) {
  // get the textarea next to the image icon that has the meta data in it
  const textarea = event.target.nextElementSibling;
  // add the text from the textarea to the clipboard
  navigator.clipboard.writeText(textarea.value);
}

function imageZoomIn(event) {
  // click event for images that creates an overlay div and houses a copy of the image showing fullscreen
  
  // get the image that was clicked on
  const image = event.target;
  // create the overlay div to black out the rest of the website
  const overlay = document.createElement("div");
  // add styling to the overlay div
  overlay.style = "position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.9); z-index: 999;	transition: all 0.2s ease-in-out;	cursor: zoom-out;"
  // create a copy of the image in the overlay div
  overlay.innerHTML = `<img src="${image.src}" style="width: 90%;	height: 90%; position: absolute;	top: 50%;	left: 50%; transform: translate(-50%, -50%);	object-fit: contain;">`;
  // add an click event to the overlay that will delete the overlay when its clicked
  overlay.addEventListener("click", function() {
		overlay.remove();
	});
  // append the overlay div to the body
  document.body.appendChild(overlay);
}

function copyToClipboard(x) {
  // used to copy prompts from .prompt files to clipboard
  navigator.clipboard.writeText(x.join(', '));
  return x;
}