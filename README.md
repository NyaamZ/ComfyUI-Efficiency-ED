✨🍬Efficiency Nodes for ComfyUI Version 2.0+ 에서 다양한 기능을 추가한 포크 버전. 원본의 설명은 https://github.com/jags111/efficiency-nodes-comfyui 를 참조하자.🍬


<b> Efficiency Nodes 💬ED
=======
### 워크플로 (EXIF 있음):
<p align="left">
  <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/a2cb4278-4294-4a16-9c07-14ae9081f1f0" width="800" style="display: inline-block;">
</p>
원본과 다르게 💬ED노드는 Context 링크를 주고 받는다.<br>

### Context:
<p align="left">
  <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/1c55eecb-7c9d-402d-bf3d-9ecb4c109d3d" width="600" style="display: inline-block;">
</p>
context를 사용해 어지럽게 널린 링크들을 위의 그림 처럼 단 한개로 정리했다!<br><br>
context는 model, clip, vae, positve 컨디셔닝, negative 컨디셔닝, 등등이 합쳐져 있는 코드 다발로 생각하면 된다.<br>
(rgthree의 커스텀 노드에서 차용)<br>
Efficiency Nodes 💬ED의 context는 rgthree의 노드가 없어도 독립적으로 작동하지만 rgthree의 노드 설치를 권장한다. 당연하지만 rgthree의 context와 호환된다.<br><br>
<details>
  <summary><b>context 간단 사용법</b></summary>
<ul>
<p align="left">
  <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/cf795977-8ab6-4646-9d28-02737122cd88" width="300" style="display: inline-block;"><br>
  context에서 특정한 요소를 추출하려면 위의 그림처럼 rgthree의 context 노드로 추출할 수 있다.</p>
<p align="left">
  <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/d82d0bd1-45fc-4f72-8cd8-15b61693db8c" width="300" style="display: inline-block;"><br>
  context에 특정한 요소를 입력하려면 위의 그림처럼 하면된다.</p>
</ul></details>

### 추가한 💬ED 노드:
<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
<details>
    <summary><b>Efficient Loader 💬ED</b></summary>
<ul>
    <p></p>
    <li>클릭 한번으로 Txt2Img, Img2Img, Inpaint 모드 설정 가능하다.<br><i>(Txt2Img로 설정시 처음에 연결된 Ksampler (Efficient) 💬ED의 denoise 값이 자동으로 1로 설정됨.)</i><br>
      <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/0f8549b8-cbe0-4662-b922-df21545e2d8f" width="250" style="display: inline-block;">
      </li>
    <li>seed, cfg, sampler, scheduler를 설정하고 <code>context</code>에 저장. 후에 Ksampler (Efficient) 💬ED등에서 그 설정값을 이용할 수 있다.</li>
    <p></p>
    <li>오른 클릭에 드롭다운 메뉴 추가.<br>
        <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/47995eca-94fb-4e52-b77b-2a53e9f292d0" width="150" style="display: inline-block;">
        <p> "🔍 View model info..."는 모델의 정보를 표시한다.<br>          
          <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/f7cf378c-cd8a-49cb-9389-5681caacf130" width="250" style="display: inline-block;"><br>
          <i>("🔍 View model info..."는 크기가 큰 모델은 해쉬값을 찾느라 '첫' 로딩이 느리다. 처음 한번은 "save as preview"를 눌러 주는걸 권장.)</i><br></p>
        <p> "📐 Aspect Ratio..."는 image_width와 image_height에 선택한 값을 입력한다.<br>
          <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/f92fdd33-ddcb-4b42-904c-4c67a52e4aa0" width="250" style="display: inline-block;"><br>
          <i>(Txt2Img 모드로 이미지를 만들 때 편리하다. ◆ 표시는 추천 해상도)</i><br></p>
    </li>
    <li>모델 선택시 프리뷰 이미지 표시<br>
        <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/9ff41533-ba10-4707-a61b-61167aea23a9" width="250" style="display: inline-block;"><br>
          <i>(이름 입력 창은 하위 폴더별로 서브메뉴가 만들어지며 "🔍 View model info..."에서 "save as preview"했던 이미지를 모델 선택시 보여준다.</i><br>
          <i>모델의 프리뷰 이미지가 있다면 이름 옆에 '*'로 표시된다.</i><br>
          <i>폴더와 모델이 함께 있을땐 유형 별로 정렬이 안되는데 그땐 폴더 이름 맨 앞에 '-'를 붙여주면 정렬이 된다.)</i><br>
    </li>
    <p></p>
    <li>Tiled VAE 인코딩<br>
        <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/b160f24f-09f6-460f-a1a4-e906077ff61b" width="300" style="display: inline-block;"><br>
          - 오른 클릭 > Property Panel에서 Use tiled VAE encode를 true로 하면 VAE 인코딩시에 Tiled VAE 인코딩을 사용할 수 있다.<br>
          - Tiled VAE 인코딩은 큰 이미지를 VRAM이 부족해도 인코딩할 수 있다. 대신 기본보다 느리다.<br>
    </li>
    <p></p>
    <li>로라, 임베딩, 컨트롤 넷 스태커를 <code>lora_stack</code>과 <code>cnet_stack</code>에 입력 가능.</li>
    <p></p>
    <li>positive와 negative 프롬프트 텍스트 박스 내장. <code>token_normalization</code>과 <code>weight_interpretation</code>에서 프롬프트 <a href="https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb">인코딩</a> 방식 설정 가능.</li>
    <p></p>
    <li>Efficient Loader 💬ED에서 context로 출력하는 값은: model, clip, vae, positive, negative, latent, images, seed, cfg, sampler, scheduler, clip_width=image_width, clip_height=image_height, text_pos_g=positive_text, text_neg_g=negative_text 이다.</li>
</ul>
</details>
<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
<details>
    <summary><b>KSampler (Efficient) 💬ED</b>, <b>KSampler TEXT (Eff.) 💬ED</b></summary>
<p></p>
- 원래 에피션트 노드에서 context를 입력 받을 수 있게 수정.<p></p>
- 이미지를 샘플링 후 context와 OUTPUT_IMAGE에 출력한다. SOURCE_IMAGE는 입력받은 이미지.<p></p>
- KSampler TEXT (Eff.) 💬ED는 배경 제작용으로 따로 프롬프트 텍스트 입력창을 추가한 버전.<br>
  (KSampler TEXT (Eff.) 💬ED가 생성하는 이미지 사이즈는 image_source_to_use로 선택에 따라 context의 이미지 또는 latent를 참조하고 텍스트 입력창의 프롬프트 텍스트는 context에 저장하지 않는다.)
<p align="left">
  <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/37ca01cb-0b8e-4e14-9d86-7dcf09c3a481" width="500">
</p>
    <p></p>
    <li>set_seed_cfg_sampler 설정으로 context에서 seed, cfg, sampler, scheduler를 가져오기 또는 내보내기가 가능함<br>
      <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/57694db3-b520-47ef-b401-8fcbfd1eb63b" width="250" style="display: inline-block;"><br>
      - from node to ctx는 현재 노드의 seed, cfg, sampler, scheduler 설정을 context에 내보내기<br>
      - from context는 context에서 seed, cfg, sampler, scheduler를 가져오기<br>
      - from node only는 현재 노드의 seed, cfg, sampler, scheduler 설정을 이용하고 context에 저장하지는 않는다.<br>
    </li>
    <p></p>
    <li>image_source_to_use 설정<br>
      <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/65cb4134-d784-4810-a56c-49b09f8bf8ef" width="250" style="display: inline-block;"><br>
      - context의 Image나 latent 중 무엇을 이미지 소스로 샘플링할까 선택하는 창이다.<br>
      - Image가 선택되면 vae decode 설정에 따라 내부에서 vae 인코딩을 하며 image_opt가 입력되면 그 이미지를 우선 사용한다.
    </li>
    <p></p>
    <li>vae decode 설정<br>
      <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/592edea3-2e16-4c29-90a3-3dd5ddd0eb63" width="250" style="display: inline-block;"><br>
      - 샘플링 후 이미지 생성을 위한 vae 디코딩/인코딩시에 무엇을 사용할지 선택하는 창이다.<br>
      - True, True(tiled), false가 있으며 기본은 True, True(tiled)는 Tiled VAE decode 사용(느리다. 대신 VRAM이 부족해도 큰 이미지 처리 가능), false는 이미지를 내보내지 않고 context에 latent만 내보낸다.
    </li>
</details>
<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
<details>
    <summary><b>Load Image 💬ED</b></summary>
<p></p>
<p align="left">
  <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/6defb14b-7492-4a75-919a-f5632bc77ec5" width="300">
</p>
- 원래 Load Image에서 프롬프트 텍스트를 출력하게 수정한 노드이다.<p></p>
<li>큐를 돌리면 아래처럼 프롬프트, seed, 이미지 사이즈가 표시된다. <br>
  <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/5b18adb0-5e8e-4cc0-963d-287cb5d19e38" width="500"><br>
  (아쉽게도 이미 설치된 노드의 프롬프트만 추출할 수 있으며, 설치되지 않은 노드는 추출하지 못한다.)<br>
</li>
</details>
<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
<details>
  <summary><b>LoRA Stacker</b></summary>
  <p></p>
  <p align="left">
  <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/857d98ec-b7f5-4957-9fc3-68a7245829cc" width="300">
  </p>
  - 최대 8개까지의 로라를 한번에 로딩할 수있는 노드이다.<p></p>
  <p></p>
  <li>Efficient Loader 💬ED와 마찬가지로 이름 입력 창은 하위 폴더별로 서브메뉴가 만들어지며 로라의 프리뷰 이미지 표시<br>
    <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/68240631-6962-4601-9f7a-2913a9eebedb" width="300"><br>
    <i>(로라의 프리뷰 이미지가 있다면 이름 옆에 '*'로 표시된다.</i><br>
    <i>폴더와 로라가 함께 있을땐 유형 별로 정렬이 안되는데 그땐 폴더 이름 맨 앞에 '-'를 붙여주면 정렬이 된다.)</i><br>
  </li>
  <p></p>
  <li>"🔍 View model info..."는 아래처럼 트리거 워드(Trained words)를 찾는데 편리하다.<br>
    <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/fe112563-4189-4d7e-aa41-72b8030fa69a" width="400">
  </li>
</details>
<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
<details>
  <p></p>
  <summary><b>Embedding Stacker 💬ED</b></summary>
  <p></p>
  <p align="left">
  <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/66ca8ba4-f6e9-4881-ba8f-e737d8609515" width="400">
  </p>
  - 임베딩 일일이 치는거 스펠링도 기억안나고 짜증나서 하나 만들었다.<br>
  <i>(기능은 단순하게 💬ED 로더 positive, negative의 맨 마지막에 임베딩 문자열을 추가해준다.</i><br>
  <i> 💬ED 로더만 사용 가능함.)</i><br>
  <p></p>
  - 로라 스태커와 동일하게 "🔍 View model info..."로 정보를 볼 수 있다.<p></p>
</details>
<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
<details>
  <p></p>
  <summary><b>Control Net Script 💬ED</b></summary>
  <p></p>
  <p align="left">
  <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/4ebd0668-f182-40a0-a882-35fb485ede5c" width="400">
  </p>
  - 컨트롤넷 스크립트. 샘플러와 연결하여 컨트롤넷을 사용하는 모드.<br>
  <i>(KSampler (Efficient) 💬ED 또는 KSampler TEXT (Eff.) 💬ED에서만 동작한다.)</i><br>
</details>
<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
<details>
  <p></p>
  <summary><b>FaceDetailer 💬ED</b></summary>
  <p></p>
  <p align="left">
  <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/0e389054-ad3e-4f02-ba2b-c1ac38af47ce" width="250">
  </p>
  - Impact pack의 FaceDetailer 애드온. Impact pack이 설치되지 않았다면 보이지 않는다.<p></p>
  - context를 입력받을 수 있게 수정한 버전.<p></p>
  - 💬ED 샘플러와 마찬가지로 set_seed_cfg_sampler 설정이 있으며, 각종 모델 로더를 통합한 노드.<p></p>
  <li>아래처럼 positive_text_opt나 negative_text_opt에 프롬프트 텍스트를 입력할 수 있다.<br>
    <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/447f8c44-75a5-4714-8b79-ad3977f58cae" width="400"><br>
    <i>(FaceDetailer 💬ED에서 눈을 더 반짝이게 하고 싶다던가 표정을 바꾸고 싶을 때 유용하다.</i><br>
    <i>프롬프트 텍스트를 입력하면 context의 프롬프트는 무시하고 입력된 프롬프트를 우선 사용한다.)</i><br>
  </li>
</details>
<details>
  <p></p>
  <summary><b>Ultimate SD Upscale 💬ED</b></summary>
  <p></p>
  <p align="left">
  <img src="https://github.com/jags111/efficiency-nodes-comfyui/assets/43065065/34fc20e4-8577-4716-9197-f63a31a6a31f" width="200">
  </p>
  - Ultimate SD Upscale의 애드온. Ultimate SD Upscale이 설치되지 않았다면 보이지 않는다.<p></p>
  - context를 입력받을 수 있게 수정한 버전.<p></p>
  - 💬ED 샘플러와 마찬가지로 set_seed_cfg_sampler 설정이 있으며, upscale 모델 로더를 통합한 노드.
  <p></p>
</details>

## Workflow Examples:

Kindly load all PNG files in same name in the (workflow driectory) to comfyUI to get all this workflows. The PNG files have the json embedded into them and are easy to drag and drop !<br>

1. HiRes-Fixing<br>
   [<img src="https://github.com/jags111/efficiency-nodes-comfyui/blob/main/workflows/HiResfix_workflow.png" width="800">](https://github.com/jags111/efficiency-nodes-comfyui/blob/main/workflows/HiResfix_workflow.png)<br>

2. SDXL Refining & **Noise Control Script**<br>
   [<img src="https://github.com/jags111/efficiency-nodes-comfyui/blob/main/workflows/SDXL_base_refine_noise_workflow.png" width="800">](https://github.com/jags111/efficiency-nodes-comfyui/blob/main/workflows/SDXL_base_refine_noise_workflow.png)<br>

3. **XY Plot**: LoRA <code>model_strength</code> vs <code>clip_strength</code><br>
   [<img src="https://github.com/jags111/efficiency-nodes-comfyui/blob/main/workflows/Eff_XYPlot%20-%20LoRA%20Model%20vs%20Clip%20Strengths01.png" width="800">](https://github.com/jags111/efficiency-nodes-comfyui/blob/main/workflows/Eff_XYPlot%20-%20LoRA%20Model%20vs%20Clip%20Strengths01.png)<br>

4. Stacking Scripts: **XY Plot** + **Noise Control** + **HiRes-Fix**<br>
   [<img src="https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/v2.0/workflows/XYPlot%20-%20Seeds%20vs%20Checkpoints%20%26%20Stacked%20Scripts.png" width="800">](https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/v2.0/workflows/XYPlot%20-%20Seeds%20vs%20Checkpoints%20%26%20Stacked%20Scripts.png)<br>

5. Stacking Scripts:  **HiRes-Fix** (with ControlNet)<br>
  [<img src="https://github.com/jags111/efficiency-nodes-comfyui/blob/main/workflows/eff_animatescriptWF001.gif" width="800">](https://github.com/jags111/efficiency-nodes-comfyui/blob/main/workflows/eff_animatescriptWF001.gif)<br>

6. SVD workflow: **Stable Video Diffusion** + *Kohya Hires** (with latent control)<br>
  <br>


### Dependencies
The python library <i><a href="https://github.com/danthedeckie/simpleeval" >simpleeval</a></i> is required to be installed if you wish to use the **Simpleeval Nodes**.
<pre>pip install simpleeval</pre>
Also can be installed with a simple pip command <br>
'pip install simpleeval'

A single file library for easily adding evaluatable expressions into python projects. Say you want to allow a user to set an alarm volume, which could depend on the time of day, alarm level, how many previous alarms had gone off, and if there is music playing at the time.

check Notes for more information.

## **Install:**
To install, drop the "_**efficiency-nodes-comfyui**_" folder into the "_**...\ComfyUI\ComfyUI\custom_nodes**_" directory and restart UI.

## Todo

[ ] Add guidance to notebook


# Comfy Resources

**Efficiency Linked Repos**
- [BlenderNeko ComfyUI_ADV_CLIP_emb](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb)  by@BlenderNeko
- [Chrisgoringe cg-noise](https://github.com/chrisgoringe/cg-noise)  by@Chrisgoringe
- [pythongosssss ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts)  by@pythongosssss
- [shiimizu ComfyUI_smZNodes](https://github.com/shiimizu/ComfyUI_smZNodes)  by@shiimizu
- [LEv145_images-grid-comfyUI-plugin](https://github.com/LEv145/images-grid-comfy-plugin))  by@LEv145
- [ltdrdata-ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) by@ltdrdata
- [pythongosssss-ComfyUI-custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) by@pythongosssss
- [RockOfFire-ComfyUI_Comfyroll_CustomNodes](https://github.com/RockOfFire/ComfyUI_Comfyroll_CustomNodes) by@RockOfFire 

**Guides**:
- [Official Examples (eng)](https://comfyanonymous.github.io/ComfyUI_examples/)- 
- [ComfyUI Community Manual (eng)](https://blenderneko.github.io/ComfyUI-docs/) by @BlenderNeko

- **Extensions and Custom Nodes**:  
- [Plugins for Comfy List (eng)](https://github.com/WASasquatch/comfyui-plugins) by @WASasquatch
- [ComfyUI tag on CivitAI (eng)](https://civitai.com/tag/comfyui)-   
- [Tomoaki's personal Wiki (jap)](https://comfyui.creamlab.net/guides/) by @tjhayasaka

  ## Support
If you create a cool image with our nodes, please show your result and message us on twitter at @jags111 or @NeuralismAI .

You can join the <a href="https://discord.gg/vNVqT82W" alt="Neuralism Discord"> NEURALISM AI DISCORD </a> or <a href="https://discord.gg/UmSd4qyh" alt =Jags AI Discord > JAGS AI DISCORD </a> 
Share your work created with this model. Exchange experiences and parameters. And see more interesting custom workflows.

Support us in Patreon for more future models and new versions of AI notebooks.
- tip me on <a href="https://www.patreon.com/jags111"> [patreon]</a>

 My buymeacoffee.com pages and links are here and if you feel you are happy with my work just buy me a coffee !

 <a href="https://www.buymeacoffee.com/jagsAI"> coffee for JAGS AI</a> 

Thank you for being awesome!

<img src = "images/ComfyUI_temp_vpose_00005_.png" width = "50%"> 

<!-- end support-pitch -->

