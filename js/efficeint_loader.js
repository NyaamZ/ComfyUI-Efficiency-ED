import {app} from "../../scripts/app.js";
import {ComfyWidgets} from "../../scripts/widgets.js";

// Displays prompt and setting on the node
app.registerExtension({
    name: "ed.efficientLoaderED",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Efficient Loader 💬ED") { //|| nodeData.name === "Eff. Loader SDXL 💬ED") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
				
				//console.log("##Efficient Loader loaded##");
				//this.setProperty("Model output only (Not included ctx)", false);
                this.setProperty("Image size sync this", true);
				this.setProperty("Image size sync MultiAreaConditioning", false);
				this.setProperty("Use tiled VAE encode", false);
				this.setProperty("Tiled VAE encode tile size", 512);
				/* this.setProperty("Kohya-block number", 3)
				this.setProperty("Kohya-downscale factor", 2.000)
				this.setProperty("Kohya-start percent", 0.000)
				this.setProperty("Kohya-end percent", 0.350 )
				this.setProperty("Kohya-downscale after skip", true)
				this.constructor["@Kohya-downscale after skip"] = { type: "boolean" };
				this.setProperty("Kohya-downscale method", "bilinear")
				this.constructor["@Kohya-downscale method"] = {
					type: "combo",
					values: ["bicubic", "nearest-exact", "bilinear", "area", "bislerp"],
				};
				this.setProperty("Kohya-upscale method", "bilinear")
				this.constructor["@Kohya-upscale method"] = {
					type: "combo",
					values: ["bicubic", "nearest-exact", "bilinear", "area", "bislerp"],
				}; */
                return result;
            };
        }
    },
});
