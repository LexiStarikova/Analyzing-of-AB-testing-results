#!/usr/bin/env python
# coding: utf-8

# ## Задание 1
# #### Представьте, что вы аналитик в компании, которая разрабатывает приложение для обработки и оформления фотографий в формате Stories (например, для дальнейшего экспорта в Instagram Stories). Был проведен A/B тест: тестовой группе предлагалась новая модель оплаты коллекций шаблонов, контрольной – старая механика. Ваша основная задача: проанализировать итоги эксперимента и решить, нужно ли выкатывать новую модель на остальных пользователей.
# 
# #### В ходе отчета обоснуйте выбор метрик, на которые вы обращаете внимание. Если различия есть, то объясните, с чем они могут быть связаны и являются ли значимыми.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from tqdm.auto import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={"figure.figsize": (14, 6)})  # width=14, #height=6


# In[2]:


# разбивка пользователей на контрольную (А) и тестовую (В) группы
groups = pd.read_csv("Проект_4_groups.csv")
groups.head()


# In[3]:


groups.value_counts(
    "group"
)  # оценим размеры выборок: тестовая группа примерно в 4 раза больше контрольной


# In[4]:


# информация о пользователях, которые посещали приложение во время эксперимента
active_users = pd.read_csv("Проект_4_active_users.csv")
active_users.head()


# In[5]:


groups.merge(active_users, how="right", on="user_id").value_counts("group")


# In[6]:


# данные о транзакциях (оплатах) пользователей приложения во время эксперимента
purchases = pd.read_csv("Проект_4_purchases.csv")
purchases.head()


# In[7]:


groups.merge(purchases, how="right", on="user_id").value_counts("group")


# #### Заметим, что в каждом датафрейме элементов в группе (B) примерно в 4 раза больше, чем в группе (А).
# #### Однако размеры группы (А) достаточно велики, чтобы выдвинуть предположение о том, что поведение наших пользователей при бОльших размерах группы будет аналогичным.
# #### В качестве ключевых метрик я буду использовать средний/медианный показатель дохода по платящим пользователям ARPPU (т.к. основной метрикой для большинства продуктов является количество денег, которое он приносит) и их конверсию в покупку (CR в событие показывает, насколько услуга актуальна и востребована).
# #### Важно отметить, что, например, метрика вовлечённости пользователей (соотношение автивных пользователей и пользователей, не посетивших приложение во время эксперимента) никак не коррелирует с проводимым тестом: неактивные пользователи попросту не могли попробовать новую модель оплаты.

# In[8]:


# Оценим медианные и средние показатели дохода с пользователей по группам.
# Заметим, что в группе (B) оба значения выросли в 1.35 и в 1.29 раза соответственно.
groups.merge(purchases, how="right", on="user_id").groupby("group", as_index=False).agg(
    {"revenue": ["median", "mean"]}
)


# In[9]:


# Проанализируем результаты в разрезах по стране, платформе и полу пользователей.
# Объединим в единый датафрейм всех активных пользователей вне зависимости от наличия оплаты:
summary = groups.merge(active_users, how="right", on="user_id").merge(
    purchases, how="left", on="user_id"
)
summary.head()


# In[10]:


sns.histplot(summary.query('group == "A"').revenue, bins=30, kde=True);


# In[11]:


sns.histplot(summary.query('group == "B"').revenue, bins=30, kde=True);


# #### Наши распределения очень ассиметричны, поэтому в качестве сравнения на первом этапе будем использовать медианные значения дохода:

# In[12]:


revenue_by_country = (
    summary.groupby(["group", "country"], as_index=False)
    .agg({"revenue": "median"})
    .pivot(index="group", columns="country", values="revenue")
)
revenue_by_country.head()


# In[13]:


revenue_by_platform = (
    summary.groupby(["group", "platform"], as_index=False)
    .agg({"revenue": "median"})
    .pivot(index="group", columns="platform", values="revenue")
)
revenue_by_platform.head()


# In[14]:


revenue_by_sex = (
    summary.groupby(["group", "sex"], as_index=False)
    .agg({"revenue": "median"})
    .pivot(index="group", columns="sex", values="revenue")
)
revenue_by_sex.head()


# #### Заметим, что в разрезе по каждому параметру медианный показатель дохода существенно возрос.
# #### Теперь проверим статистическую значимость полученных результатов:

# In[15]:


# И для данного этапа анализа отберём только пользователей, совершивших оплату:
summary_with_revenue = summary[summary["revenue"].notna()]
summary_with_revenue.head()


# In[16]:


summary_with_revenue.revenue.isna().sum()


# In[17]:


# Еще раз взглянем на графики распределения показателя дохода по группам:
sns.histplot(summary_with_revenue.query('group == "A"').revenue, bins=30, kde=True);


# In[18]:


sns.histplot(summary_with_revenue.query('group == "B"').revenue, bins=30, kde=True);


# #### Мы имеем дело с ненормальным распределением (становится очевидно по графику), и это подтверждается тестом Шапиро-Уилка:

# In[19]:


# p-value << 0.05, значит, распределение ненормальное
stats.shapiro(summary_with_revenue.query('group == "A"').revenue)


# In[20]:


stats.shapiro(summary_with_revenue.query('group == "B"').revenue)


# ####  Для проведения тестирования на статистическую значимость результата используем бутстрап: он позволяет многократно извлекать подвыборки из выборки, полученной в рамках эксперимента; в полученных подвыборках считаются статистики (среднее, медиана и т.п.), из которых можно получить ее распределение и взять доверительный интервал. Данный метод является методом непараметрической статистики.
# 

# In[21]:


from scipy.stats import norm


# In[22]:


# Объявим функцию, которая позволит проверять гипотезы с помощью бутстрапа


def get_bootstrap(
    data_column_1,  # числовые значения первой выборки
    data_column_2,  # числовые значения второй выборки
    boot_it=1000,  # количество бутстрэп-подвыборок
    statistic=np.mean,  # интересующая нас статистика
    bootstrap_conf_level=0.95,  # уровень значимости
):
    boot_len = max([len(data_column_1), len(data_column_2)])
    boot_data = []
    for i in tqdm(range(boot_it)):  # извлекаем подвыборки
        samples_1 = data_column_1.sample(
            boot_len, replace=True  # параметр возвращения
        ).values

        samples_2 = data_column_2.sample(boot_len, replace=True).values

        boot_data.append(
            statistic(samples_1 - samples_2)
        )  # mean() - применяем статистику

    pd_boot_data = pd.DataFrame(boot_data)

    left_quant = (1 - bootstrap_conf_level) / 2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    ci = pd_boot_data.quantile([left_quant, right_quant])

    p_1 = norm.cdf(x=0, loc=np.mean(boot_data), scale=np.std(boot_data))
    p_2 = norm.cdf(x=0, loc=-np.mean(boot_data), scale=np.std(boot_data))
    p_value = min(p_1, p_2) * 2

    # Визуализация
    plt.hist(pd_boot_data[0], bins=50)

    plt.style.use("ggplot")
    plt.vlines(ci, ymin=0, ymax=50, linestyle="--")
    plt.xlabel("boot_data")
    plt.ylabel("frequency")
    plt.title("Histogram of boot_data")
    plt.show()

    return {"boot_data": boot_data, "ci": ci, "p_value": p_value}


# In[23]:


# Применим бутстрап на наших выборках. В качестве статистики используем средние значения.
booted_data = get_bootstrap(
    summary_with_revenue.query('group == "A"').revenue,
    summary_with_revenue.query('group == "B"').revenue,
)


# In[24]:


# Получаем p-value << 0.05, что говорит нам о статистической значимости результата.
booted_data["p_value"]


# #### Сделаем повторную проверку - проверим гипотезу на непараметрическом тесте Манна-Уитни: требования для его проведения минимальны, и численность сравниваемых групп может быть не одинаковой.

# In[25]:


# Снова получаем p-value << 0.05 и подтверждаем полученный результат.
stats.mannwhitneyu(
    summary_with_revenue.query('group == "A"').revenue,
    summary_with_revenue.query('group == "B"').revenue,
)


# #### В ходе проведённого выше этапа исследования мы выявили статистически значимые различия между показателями ARPPU в контрольной и тестовой выборках. Этот вывод был подстверждён двумя различными по алгоритму проведения тестами.
# #### Полученная статистическая значимость различий означает, что наше нововведение действительно оказало влияние на метрику ARPPU и уже имеет смысл рассмотреть полноценное внедрение новой модели оплаты.

# #### Теперь обратим внимание на вторую, не менее важную для нас метрику - конверсию.

# In[26]:


# Нас больше не интересует размер дохода с пользователя:
# теперь мы ориентируемся на бинарную переменную наличия оплаты.
# Используем уже сформированный ранее датафрейм, в котором
# отобраны только активные пользователи вне зависимости от наличия оплаты.

summary["revenue"] = summary.revenue.apply(lambda x: 0 if np.isnan(x) else 1)
summary.head()


# #### Эта метрика интересует нас в том числе и в качестве "подстраховки" на случай, если модель оплаты оказалась непригодной к использованию в каких-либо сегментах пользователей.

# In[27]:


# Чтобы не повторять один и тот же код для расчётов в разрезах, зададим функцию:


def get_conversion_rate(data, separation_parameters):
    # data has to contain the columns 'revenue' with binary values (0, 1)
    # separation_parameters has to be a string or a list

    number_of_conv_users = (
        data.groupby(separation_parameters, as_index=False)
        .agg({"revenue": "sum"})
        .rename(columns={"revenue": "converted_users"})
    )  # count the number of converted users

    number_of_users = (
        data.groupby(separation_parameters, as_index=False)
        .agg({"revenue": "count"})
        .rename(columns={"revenue": "all_users"})
    )  # count the number of all users

    cr = number_of_users.merge(number_of_conv_users, on=separation_parameters)
    cr["conversion_rate"] = cr.converted_users / cr.all_users

    return cr


# In[28]:


get_conversion_rate(summary, "group")


# In[29]:


get_conversion_rate(summary, ["group", "country"]).pivot(
    index="group", columns="country", values="conversion_rate"
)


# In[30]:


get_conversion_rate(summary, ["group", "platform"]).pivot(
    index="group", columns="platform", values="conversion_rate"
)


# In[31]:


get_conversion_rate(summary, ["group", "sex"]).pivot(
    index="group", columns="sex", values="conversion_rate"
)


# #### Заметим, что в каждом разрезе в группе (В) CR стало на десятые доли процента ниже, чем в группе (А).
# #### Проверим статистическую значимость результата на всём объеме данных: здесь мы работаем с качественной (бинарной) переменной, поэтому для оценки значимости различий между группами используем критерий согласия Пирсона или же хи-квадрат:

# In[32]:


# Подготовим перекрестную таблицу:
pd.crosstab(summary.revenue, summary.group)


# In[41]:


# p-value > 0.05, значит, мы НЕ можем отклонить нулевую гипотезу 
# и переменные не имеют существенной связи.
# Значит, наше нововведение с большей вероятностью
# не оказывает влияние на метрику конверсии.
stat, p, dof, expected = stats.chi2_contingency(
    pd.crosstab(summary.revenue, summary.group)
)
print("p-value =", p)


# #### Результаты проверки значимости различий между двумя группами показали, что изменения не оказали статистически значимого влияния на метику конверсии. Значит, выявленные ранее различия, вероятнее всего, являются случайными, и наш эксперимент не понёс за собой никаких статистически значимых изменений метрики CR в тестовой группе.

# #### В результате анализа итогов эксперимента по внедрению новой модели оплаты коллекций шаблонов было выявлено, что нововведение оказывает статистически значимое влияние на метрику ARPPU - показатель существенно возрос во всех сегментах. В то же время эксперимент никак не проявил себя с точки зрения метрики конверсии в оплату, откуда следует отстуствие явных сбоев модели.
# #### Основываясь на результатах проведенного тестирования, мы можем сделать вывод о целесообразности внедрения новой модели оплаты коллекций шаблонов на всех остальных пользователей.

# ## Задание 2
# #### Одной из основных задач аналитика является не только построение моделей, но и создание дашбордов, которые позволяют отслеживать изменения метрик и принимать на их основе оптимальные решения. Ваш руководитель хочет узнать, как обстоят дела с использованием приложения и вовлечённостью пользователей, и очень просит спроектировать ему дашборд.

# In[34]:


# Преобразуем имеющиеся данные в удобный для визуализации результатов теста файл


# In[35]:


groups.head()


# In[36]:


active_users["active_user"] = 1
active_users.head()


# In[37]:


purchases["converted_user"] = 1
purchases.head()


# In[38]:


df = groups.merge(active_users, how="left", on="user_id").merge(
    purchases, how="left", on="user_id"
)
df["active_user"].fillna(0, inplace=True)
df["converted_user"].fillna(0, inplace=True)
df = df.astype({"active_user": "int", "converted_user": "int"})
df.head()


# In[39]:


# Теперь запишем датафрейм в csv файл для работы в Tableau

"""
with open('data.csv', 'w') as f:
    f.write(df.to_csv(index=True, header=True))
""";


# #### Ссылка на дашборд: https://public.tableau.com/app/profile/starikova.alexandra/viz/ABTestResults/ABTestResults#1
