library(ggplot2)
library(dplyr)
library(readr)
library(stringr)
library(forcats)
library(sf)

models_df <- read_csv("data/wandb_export_2026-05-20T17_52_38.601+01_00.csv") |> 
  mutate(model_type = case_when(str_detect(arch, "convnext") ~ "ConvNext",
                                str_detect(arch, "densenet") ~ "DenseNet",
                                str_detect(arch, "resnet") ~ "ResNet",
                                str_detect(arch, "mobilenet") ~ "MobileNet",
                                str_detect(arch, "efficientnet") ~ "EfficientNet")) |> 
  mutate(embedding = case_when(embedding == "GeoTessera_v1.1" ~ "Tessera v1.1", 
                               embedding == "GeoTessera" ~ "Tessera",
                               embedding == "AlphaEarthCoop" ~ "AlphaEarth", 
                               embedding == "EmbeddedSeamless" ~ "Embedded Seamless Data")) |> 
  mutate(embedding = fct_relevel(embedding, rev(c("Tessera v1.1", "Tessera",
                                              "AlphaEarth", "Embedded Seamless Data")))) |> 
  mutate(preset = fct_relevel(preset, c("small", "medium", "large")))
models_df |> 
  filter(model_type %in% c("ConvNext", "DenseNet", "ResNet", "MobileNet", "EfficientNet")) |>
ggplot() +
  aes(y = embedding, x = test_kappa, fill = preset) +
  geom_bar(stat = 'identity', position=position_dodge()) +
  scale_x_continuous(breaks = c(0, .5, .7, .8, .9)) +
  facet_wrap(~model_type, ncol = 1, strip.position = 'right') +
  labs(x = "Cohen's Kappa Score", y = NULL, fill = " Model Size") +
  theme_minimal() + 
  theme(legend.position = 'bottom', 
  axis.text.y = element_text(size = 12),
        strip.text = element_text(size = 12))

lcz_colours = c("#8c0000",
"#d10000",
"#ff0000",
"#bf4d00",
"#ff6600",
"#ff9955",
"#faee05",
"#bcbcbc",
"#ffccaa",
"#555555",
"#006a00",
"#00aa00",
"#648525",
"#b9db79",
"#000000",
"#fbf7ae",
"#6a6aff")

patches_gdf <- sf::read_sf(paste0(Sys.getenv("DATA_DIR"), 
                           "/input/So2Sat-LCZ42/v4/patches_reference_rxr.gpkg")) |> 
  mutate(dataset = fct_relevel(dataset, c("training", "validation", "testing")))
patches_gdf |> 
ggplot() +
  aes(x = factor(LCZ_class), fill = factor(LCZ_class)) +
  geom_bar() + 
  scale_fill_manual(values = lcz_colours) +
  facet_wrap(~dataset, ncol = 1, scales = "free") +
  labs(x = "LCZ Class", y = "Frequency") +
  theme_minimal() + theme(legend.position = "none")
