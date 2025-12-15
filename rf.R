library(tidyverse)
library(jsonlite)
library(tidytext)
library(tidymodels)

tidymodels_prefer()
set.seed(123)

#-----------------------------
# 1. Read JSON and tidy
#-----------------------------

train_raw <- read_file("~/Downloads/whats-cooking/train.json.zip") %>%
  fromJSON()

test_raw <- read_file("~/Downloads/whats-cooking/test.json.zip") %>%
  fromJSON()

json_to_tidy <- function(dat) {
  dat %>%
    as_tibble() %>%
    unnest_longer(ingredients) %>%
    rename(ingredient = ingredients)
}

train_tidy <- json_to_tidy(train_raw)
test_tidy  <- json_to_tidy(test_raw)

#-----------------------------
# 2. Counts per recipe x ingredient
#-----------------------------

train_counts <- train_tidy %>%
  count(id, cuisine, ingredient, name = "n")

test_counts <- test_tidy %>%
  count(id, ingredient, name = "n")

#-----------------------------
# 3. Top ingredients by total count
#-----------------------------

n_top_ingredients <- 800  # was 500

top_ingredients <- train_counts %>%
  count(ingredient, wt = n, name = "total_n") %>%
  slice_max(total_n, n = n_top_ingredients) %>%
  pull(ingredient)
#-----------------------------
# 4. TF–IDF on training (and IDF lookup)
#-----------------------------

train_tfidf_long <- train_counts %>%
  filter(ingredient %in% top_ingredients) %>%
  bind_tf_idf(term = ingredient,
              document = id,
              n = n) %>%
  select(id, cuisine, ingredient, tf, idf, tf_idf)

idf_lookup <- train_tfidf_long %>%
  distinct(ingredient, idf)

# For modeling we only need tf_idf:
train_tfidf_for_model <- train_tfidf_long %>%
  select(id, cuisine, ingredient, tf_idf)

#-----------------------------
# 5. TF–IDF on test using training IDF
#-----------------------------

test_doc_totals <- test_counts %>%
  group_by(id) %>%
  summarise(doc_total = sum(n), .groups = "drop")

test_tfidf_long <- test_counts %>%
  filter(ingredient %in% top_ingredients) %>%
  left_join(test_doc_totals, by = "id") %>%
  left_join(idf_lookup, by = "ingredient") %>%
  mutate(
    tf     = n / doc_total,
    tf_idf = tf * idf
  ) %>%
  select(id, ingredient, tf_idf)

#-----------------------------
# 6. Wide TF–IDF matrices
#-----------------------------

train_tfidf_wide <- train_tfidf_for_model %>%
  pivot_wider(
    id_cols     = c(id, cuisine),
    names_from  = ingredient,
    values_from = tf_idf,
    values_fill = 0
  )

test_tfidf_wide <- test_tfidf_long %>%
  pivot_wider(
    id_cols     = id,
    names_from  = ingredient,
    values_from = tf_idf,
    values_fill = 0
  )

#-----------------------------
# 7. Re-attach IDs and fill any remaining NAs
#-----------------------------

train_ids <- tibble(
  id      = train_raw$id,
  cuisine = train_raw$cuisine
)

train_tfidf_wide <- train_ids %>%
  left_join(train_tfidf_wide, by = c("id", "cuisine")) %>%
  mutate(across(-c(id, cuisine), ~ replace_na(., 0)))

test_ids <- tibble(id = test_raw$id)

test_tfidf_wide <- test_ids %>%
  left_join(test_tfidf_wide, by = "id") %>%
  mutate(across(-id, ~ replace_na(., 0)))

#-----------------------------
# 8. Final modeling data
#-----------------------------

train_model_df <- train_tfidf_wide %>%
  select(-id) %>%
  mutate(cuisine = factor(cuisine))

test_model_df <- test_tfidf_wide %>%
  select(-id)

#-----------------------------
# 9. Recipe + model + workflow
#-----------------------------

whats_recipe <- recipe(cuisine ~ ., data = train_model_df) %>%
  step_zv(all_predictors())

rf_mod <- rand_forest(
  trees = 300,  # reduced from 500 (usually similar accuracy, a bit faster)
  mtry  = floor(sqrt(ncol(train_model_df) - 1)),
  min_n = 5
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(whats_recipe)

rf_fit <- rf_wf %>%
  fit(data = train_model_df)

#-----------------------------
# 10. Predictions + submission
#-----------------------------

test_preds <- predict(rf_fit, new_data = test_model_df)

submission <- test_tfidf_wide %>%
  select(id) %>%
  bind_cols(test_preds) %>%
  rename(cuisine = .pred_class)

write_csv(
  submission,
  "~/Downloads/whatscooking_tfidf_rf.csv"
)