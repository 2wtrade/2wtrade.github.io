
library(data.table)
library(fst)
library(ggplot2)
library(knitr)
knitr::opts_chunk$set(echo = TRUE, warning = FALSE)
if (!dir.exists("output")) dir.create("output")



dt <- fread("data/orders-answer.csv", 
            header = FALSE,
            col.names = c("lead_create","lead_id","order_id","approved_at",
                          "called_at","webmaster_identifier","product",
                          "operator_login","operator_gender","operator_age",
                          "operator_country","lead_country","answer",
                          "postsale","parent_order_id"),
            na.strings = c("", "\\N"))

# Правильный тип для дат
dt[, `:=`(lead_create = as.POSIXct(lead_create, 
                                   format = "%Y-%m-%d %H:%M:%S",
                                   optional = TRUE),
          approved_at = as.POSIXct(approved_at, 
                                   format = "%Y-%m-%d %H:%M:%S",
                                   optional = TRUE),
          called_at = as.POSIXct(called_at, 
                                 format = "%Y-%m-%d %H:%M:%S",
                                 optional = TRUE))]

# Индикатор того, что заказ был подтвержден
dt[, is_approved := 0]
dt[!is.na(approved_at), is_approved := 1]

# Сохраняем в бинарном формате для более быстрого чтения
# Даты сохраняются в POSIXct
write_fst(dt, "dt.fst")



dt <- read_fst("data/dt.fst", as.data.table = TRUE)


'Отберем наблюдения, для которых указана дата звонка оператора. 
Все остальные клиенты подтвердили или не подтвердили заказ 
самостоятельно и с точки зрения оценки работы операторов 
интереса не представляют.'

dt <- dt[!is.na(called_at), ]


# Картинка

# minute_from_create
dt[, minute_from_create := as.numeric(difftime(called_at,  
                                               lead_create, 
                                               units = "mins"))]
dt[, minute_from_create := round(minute_from_create, -1)]

dt_inno_gialuron <- dt[product == "Inno Gialuron" & 
                         minute_from_create < 1300]
dt_inno_gialuron[, approved_percent := mean(is_approved) * 100, 
                 by = .(minute_from_create, lead_country)]
dt_inno_gialuron <- dt_inno_gialuron[approved_percent < 60]
dt_inno_gialuron[, total_calls := .N, 
                 by = .(minute_from_create, lead_country)]
dt_inno_gialuron <- unique(dt_inno_gialuron[, .(minute_from_create,
                                                approved_percent,
                                                total_calls,
                                                lead_country)])
dt_inno_gialuron[, lead_country := factor(lead_country)]

ggplot(dt_inno_gialuron, aes(minute_from_create, approved_percent,
                             size = total_calls, colour = lead_country)) +
  geom_point(alpha = 0.5)

#Подсчитаем, сколько звонков увенчались подтверждением заказа:
  

table(dt[, is_approved])


#Заказ был подтвержден после 9.8% всех звонков.

#Посмотрим, сколько операторов участвует в прозвоне одного лида. 


dt[, n_ops := uniqueN(operator_login), by = lead_id]
table(dt[, n_ops])



# Убираем строки без date_create
dt <- dt[!is.na(lead_create), ]

# Кодируем NA и "" в operator_gender как отдельную категорию
dt[is.na(operator_gender) | operator_gender == "", 
   operator_gender := "unknown"]

dt[, unique(operator_age)]
#  [1]  33  32  30  43  22  45  21  40  NA  27  20  46  37  26  50  23  42  31 -71  25  24  18  35
# [24]  29  19  38  28  56  36  34  48  57  39  59  41  47  51  52  58  44   0  53

# -71 и 0 в operator_age меняем на NA
dt[operator_age == -71 | operator_age == 0, operator_age := NA]

# Сортируем таблицу по lead_id, order_id и по дате звонка 
# внутри каждого заказа каждого лида
setkey(dt, lead_id, order_id, called_at)

# Убираем строки, для которых called_at предшествует lead_create
dt <- dt[lead_create < called_at]

# Добавляем количество звонков, предшествующих данному в пределах заказа
dt[, n_calls_before := dt[, .SD[, .I], by = .(lead_id, order_id)]$V1 - 1]

# Добавляем логарифм времени в часах от момента регистрации до звонка
dt[, time_diff := log(as.numeric(difftime(called_at,  
                                          lead_create, 
                                          units = "hours")))]

# Добавляем час звонка и час регистрации 
# (время суток может быть важным предиктором)
dt[, `:=`(hour_create = hour(lead_create),
          hour_call = hour(called_at))]

# Удаляем пробелы в product
dt[, product := gsub(pattern = "\\s", replacement = "", product)]



## Модель для предсказания подтверждения 


dt_ml <- dt[, .(product, operator_login, webmaster_identifier,
                operator_gender, operator_age, 
                operator_country, n_calls_before, 
                time_diff, hour_create, 
                hour_call, is_approved)]
rm(dt)
# Удаляем строки с NA в webmaster_identifier
dt_ml <- dt_ml[!is.na(webmaster_identifier)]
factor_cols <- c("product", "operator_login", "webmaster_identifier", 
                 "operator_gender", "operator_country")
dt_ml[, (factor_cols) := lapply(.SD, as.factor), .SDcols = factor_cols]
dt_ml[, is_approved := factor(is_approved,
                              levels = c(1, 0),
                              labels = c("approved", "not_approved"))]



library(rsample)
library(recipes)
library(embed)
library(purrr)
library(yardstick)
library(glmnet)
library(ranger)


#Разобьем весь набор данных на обучающий и тестовый в пропорции 80/20:
  

set.seed(42)
data_split <- initial_split(dt_ml, 
                            prop = 0.8, 
                            strata = "is_approved")
train_data <- training(data_split)
test_data <- testing(data_split)


#Создадим схему перекрестной проверки со стратифицированной по целевой переменной с разбивкой на 5 блоков:
  

set.seed(42)
cv_splits <- vfold_cv(train_data, v = 5, strata = "is_approved")


#Строить будем модель логистической регрессии (пакет **glmnet**), проверяя три значения гиперпараметра `lambda` (сила регуляризции) и три значения гиперпараметра `alpha` (пропорция l1- и l2-штрафов).


param_grid <- expand.grid(imp_strategy = "median_imp",
                          alpha = c(0, 0.5, 1),
                          lambda = c(0.1, 0.01, 0.001)
)


#Нам понадобится функция, создающая “рецепт”, которая будет параметризована по типу импутации:
  

create_recipe <- function(data = train_data,
                          imp_strategy = "mean_imp") {
  
  imp_methods <- list(mean_imp = step_meanimpute, 
                      median_imp = step_medianimpute)
  
  rec_obj <- recipe(is_approved ~ ., data = data) %>%
    # Балансировка классов
    step_downsample(is_approved) %>%
    # Импутация средним или медианой для operator_age
    imp_methods[[imp_strategy]](operator_age) %>%
    # WOE для operator_login и webmaster_identifier
    step_woe(operator_login, outcome = is_approved) %>%
    step_woe(webmaster_identifier, outcome = is_approved) %>%
    # Преобразование Йео-Джонсона для всех количественных
    step_YeoJohnson(operator_age, n_calls_before, time_diff, 
                    hour_create, hour_call, 
                    woe_operator_login, woe_webmaster_identifier) %>%
    # Стандартизация
    step_center(operator_age, n_calls_before, time_diff, 
                hour_create, hour_call, 
                woe_operator_login, woe_webmaster_identifier) %>%
    step_scale(operator_age, n_calls_before, time_diff, 
               woe_operator_login, woe_webmaster_identifier, 
               hour_create, hour_call) %>%
    # one-hot для product, operator_gender, operator_country
    step_dummy(product, operator_gender, operator_country, 
               one_hot = TRUE) 
  return(rec_obj)
}


#Функция для обучения модели, возвращающая значение выбранной метрики (в данном случае accuracy):
  

logreg_fit_pred <- function(split, 
                            data = train_data,
                            imp_strategy = "mean_imp", 
                            lambda = 0.1, 
                            alpha = 0) {
  
  rec_obj <- create_recipe(data = data, 
                           imp_strategy = imp_strategy)
  rec_obj <- prepper(split_obj = split, rec_obj, retain = TRUE)
  
  train <- juice(rec_obj)
  val <- bake(rec_obj, 
              new_data = assessment(split),
              everything())
  
  train_x <- 
    train %>%
    select(-is_approved) %>%
    as.matrix()
  train_y <- train$is_approved
  
  val_x <- val %>%
    select(-is_approved) %>%
    as.matrix()
  val_y <- val$is_approved
  
  model <- glmnet(x = train_x, 
                  y = train_y, 
                  family = "binomial",
                  lambda = lambda,
                  alpha = alpha)
  
  preds <- predict(model, 
                   type = "response",
                   newx = val_x)
  # for a factor, the last level in alphabetical order is the target class
  #preds <- ifelse(preds > 0.5, 0, 1)
  #preds <- factor(preds, levels = c(1, 0), 
  #                labels = c("approved", "not_approved"))
  preds_auc <- data.frame(truth = val_y, predicted = preds[, 1])
  preds_conf <- data.frame(truth = val_y, 
                           predicted = factor(ifelse(preds[, 1] > 0.5, 0, 1), 
                                              levels = c(1, 0),
                                              labels = c("approved",
                                                         "not_approved")))
  
  res1 <- roc_auc(preds_auc, 
                  truth, 
                  predicted)
  res2 <- conf_mat(preds_conf, 
                   truth, 
                   predicted)
  
  return(list(auc = res1, confusion_matrix = res2))
}

logreg_fit_pred(cv_splits$splits[[1]], data = train_data)


#Функция, обучающая модели для всех комбинаций значений гиперпараметров:
  

across_grid <- function(split, grid) {
  metrics <- lapply(seq_len(nrow(grid)),
                    function(i) logreg_fit_pred(split, 
                                                data = train_data,
                                                grid[i, "imp_strategy"],
                                                grid[i, "lambda"],
                                                grid[i, "alpha"]))
  result <- do.call(rbind, metrics)
  return(cbind(grid, result))
}

across_grid(cv_splits$splits[[1]], param_grid)


#Остается лишь выполнить собственно перекрестную проверку:
  

result <- lapply(cv_splits$splits, across_grid, grid = param_grid)

result <- data.table::rbindlist(result, idcol = "split")
result[, auc := result[, rbindlist(auc)][, .estimate]]
# result[, auc := sapply(auc, "[[", ".estimate")]
result[, .(roc_auc = mean(auc)), 
       by = .(imp_strategy, lambda, alpha)
       ][
         order(roc_auc, decreasing = TRUE), ]


#Использовали балансировку классов и WOE-биннинг для `operator_login` и `webmaster_identifier`. ROC-AUC на уровне  0.76-0.77, в то время как предсказание `"not_approved"` для всех наблюдений дает значение 0.5. 

##  Дополнение - обучение и сохранение лучшей модели

#Обучим модель с одним из наиболее оптимальных наборов гиперпараметров на всей обучающей и проверим на тестовой выборке.


rec_obj <- create_recipe(data = train_data, 
                         imp_strategy = "mean_imp")
rec_obj <- prep(rec_obj, training = train_data, retain = TRUE)

train <- juice(rec_obj)
train_x <- 
  train %>%
  select(-is_approved) %>%
  as.matrix()
train_y <- train$is_approved

test <- bake(rec_obj, 
             new_data = test_data,
             everything())

test_x <- test %>%
  select(-is_approved) %>%
  as.matrix()
test_y <- test$is_approved

glmnet_model <- glmnet(x = train_x, 
                       y = train_y, 
                       family = "binomial",
                       lambda = 0.001,
                       alpha = 0.5)

# Сохранение и загрузка модели
save(glmnet_model, file = "output/glmnet_model.RData")
load("output/glmnet_model.RData")  

preds <- predict(glmnet_model, 
                 type = "response",
                 newx = test_x)

preds_auc <- data.frame(truth = test_y, predicted = preds[, 1])
preds_conf <- data.frame(truth = test_y, 
                         predicted = factor(ifelse(preds[, 1] > 0.5, 0, 1), 
                                            levels = c(1, 0),
                                            labels = c("approved",
                                                       "not_approved")))

res1 <- roc_auc(preds_auc, 
                truth, 
                predicted)
res2 <- conf_mat(preds_conf, 
                 truth, 
                 predicted)

list(auc = res1, confusion_matrix = res2)



# Predict for new data


# Пусть пришел клиент с заказом "ChocolateSlim" в промежуток от 14 до 15 часов,
# звонок предполагается через 20 минут (0.33 ч.), что соответствует времени от 15
# до 16 часов (hour_call = 15)
test_data <- data.table(product = "ChocolateSlim",
                        n_calls_before = 0,
                        time_diff = log(0.33),
                        hour_create = 14,
                        hour_call = 15)

# Добавляем данные по всем операторам в комбинации с веб-мастерами
tmp <- unique(train_data [, .(operator_login, operator_gender,
                              webmaster_identifier,
                              operator_age, operator_country)])
tmp[, product := "ChocolateSlim"]
test_data <- test_data[tmp, on = "product"]
test_data[, product := factor(product)]

test <- bake(rec_obj, 
             new_data = test_data,
             everything())
setDT(test)
test <- test[!is.na(woe_operator_login) & !is.na(woe_webmaster_identifier),
             -"is_approved"]
test_x <- as.matrix(test)

preds <- predict(glmnet_model, 
                 type = "response",
                 newx = test_x)

test[, pred_prob := preds[, 1]]
test[order(pred_prob, decreasing = TRUE)]
str(test)






#  CI
dt<- fread("orders-answer.csv", 
            header = FALSE,
            col.names = c("lead_create","lead_id","order_id","approved_at",
                          "called_at","webmaster_identifier","product",
                          "operator_login","operator_gender","operator_age",
                          "operator_country","lead_country","answer",
                          "postsale","parent_order_id"),
            na.strings = c("", "\\N"))

# Правильный тип для дат
dt[, `:=`(lead_create = as.POSIXct(lead_create, 
                                   format = "%Y-%m-%d %H:%M:%S",
                                   optional = TRUE),
          approved_at = as.POSIXct(approved_at, 
                                   format = "%Y-%m-%d %H:%M:%S",
                                   optional = TRUE),
          called_at = as.POSIXct(called_at, 
                                 format = "%Y-%m-%d %H:%M:%S",
                                 optional = TRUE))]

# Индикатор того, что заказ был подтвержден
dt[, is_approved := 0]
dt[!is.na(approved_at), is_approved := 1]

# Сохраняем в бинарном формате для более быстрого чтения
# Даты сохраняются в POSIXct
write_fst(dt, "dt.fst")



dt <- read_fst("dt.fst", as.data.table = TRUE)
dt1=dt
View(dt1)


dt <- dt[!is.na(called_at), ]



table(dt[, is_approved])



dt[, n_ops := uniqueN(operator_login), by = lead_id]



# Убираем строки без date_create
dt <- dt[!is.na(lead_create), ]

# Кодируем NA и "" в operator_gender как отдельную категорию
dt[is.na(operator_gender) | operator_gender == "", 
   operator_gender := "unknown"]

dt[, unique(operator_age)]
#  [1]  33  32  30  43  22  45  21  40  NA  27  20  46  37  26  50  23  42  31 -71  25  24  18  35
# [24]  29  19  38  28  56  36  34  48  57  39  59  41  47  51  52  58  44   0  53

# -71 и 0 в operator_age меняем на NA
dt[operator_age == -71 | operator_age == 0, operator_age := NA]

# Сортируем таблицу по lead_id, order_id и по дате звонка 
# внутри каждого заказа каждого лида
setkey(dt, lead_id, order_id, called_at)

# Убираем строки, для которых called_at предшествует lead_create
dt <- dt[lead_create < called_at]

# Добавляем время до звонка в часах
dt[, time_diff := as.numeric(difftime(called_at,  
                                      lead_create, 
                                      units = "hours"))]

# Удаляем пробелы в product
dt[, product := gsub(pattern = "\\s", replacement = "", product)]



# Количество звонков, совершенных каждым оператором 
# с группировкой по товару+стране лида+вебмастеру
dt[, 
   n_calls := .N, 
   by = .(operator_login, product, lead_country, webmaster_identifier)]
# Количество звонков, закончившихся подтверждением заказа
dt[, 
   n_approves := sum(is_approved), 
   by = .(operator_login, product, lead_country, webmaster_identifier)]
# Вероятность подтверждения после звонка
dt[, prob_approve := n_approves / n_calls]

res <- unique(dt[, .(operator_login, product, lead_country, 
                     webmaster_identifier, n_calls, n_approves, 
                     prob_approve)])
res <- res[order(prob_approve, decreasing = TRUE)]   

# Доверительные интервалы для вероятности
f_lo <- function(n_approves, n_calls) {
  binom.test(n_approves, n_calls, 0.5, alternative="two.sided")$conf.int[[1]]
}
f_up <- function(n_approves, n_calls) {
  binom.test(n_approves, n_calls, 0.5, alternative="two.sided")$conf.int[[2]]
}
res[, prob_ci_lo := purrr::map2_dbl(n_approves, n_calls, f_lo)]
res[, prob_ci_up := purrr::map2_dbl(n_approves, n_calls, f_up)]
res1=res
View(res1)



# Разбиваем на интервалы
dt[, 
   intervals := cut(time_diff, c(0, 0.5, 1:9, seq(10, 100, 10), 
                                 max(time_diff)))]

# Количество звонков, совершенных каждым оператором 
# с группировкой по товару+стране лида+вебмастеру+интервалу
dt[, 
   n_calls := .N, 
   by = .(operator_login, product, lead_country, 
          webmaster_identifier, intervals)]
# Количество звонков, закончившихся подтверждением заказа
dt[, 
   n_approves := sum(is_approved), 
   by = .(operator_login, product, lead_country, 
          webmaster_identifier, intervals)]

dt[, prob_approve := n_approves / n_calls]

dt[, mean_n_calls := n_calls / n_approves]
dt2=dt
res2=res
res2 <- unique(dt[, .(operator_login, product, lead_country, 
                     webmaster_identifier, intervals,
                     n_calls, n_approves, prob_approve)])
res2 <- res2[order(prob_approve, decreasing = TRUE)]   

View(res2)
f_lo <- function(n_approves, n_calls) {
  binom.test(n_approves, n_calls, 0.5, alternative="two.sided")$conf.int[[1]]
}
f_up <- function(n_approves, n_calls) {
  binom.test(n_approves, n_calls, 0.5, alternative="two.sided")$conf.int[[2]]
}
res2[, prob_ci_lo := purrr::map2_dbl(n_approves, n_calls, f_lo)]
res2[, prob_ci_up := purrr::map2_dbl(n_approves, n_calls, f_up)]
View(res2)
# Для строк с prob_approve == 1 или prob_approve == 0

res2[, operator_approved := purrr::map2_dbl(prob_ci_lo, prob_ci_up, 
                                           function(x, y) runif(1, x, y))]








dt <- fread("orders-answer.csv", 
            header = FALSE,
            col.names = c("lead_create","lead_id","order_id","approved_at",
                          "called_at","webmaster_identifier","product",
                          "operator_login","operator_gender","operator_age",
                          "operator_country","lead_country","answer",
                          "postsale","parent_order_id"),
            na.strings = c("", "\\N"))

# Удаляем пробелы в product
dt[, product := gsub(pattern = "\\s", replacement = "", product)]

prob_grouped_intervals <- res2
View(prob_grouped_intervals)
# Используем 1 строку c ChocolateSlim для примера
newdata <- dt[product == "ChocolateSlim", ][1, ]

res <- prob_grouped_intervals[newdata, 
                              on = c("product", 
                                     "lead_country", 
                                     "webmaster_identifier"),
                              ]


res$prob_approve<-NULL
res$prob_ci_lo<-NULL
res$prob_ci_up<-NULL
res$approved_at<-NULL
res$called_at<-NULL
res$i.operator_login<-NULL
res$answer<-NULL
res$postsale<-NULL
res$parent_order_id<-NULL



fwrite(res, "output/result.csv")
