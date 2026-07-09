## საერთო Preprocessing Pipeline

ყველა მოდელისთვის გამოვიყენეთ საერთო preprocessing pipeline (`preprocessing.py`), რათა მონაცემების გაწმენდა, გაერთიანება და დროითი სტრუქტურის მომზადება ერთნაირად გაკეთებულიყო. ეს განსაკუთრებით მნიშვნელოვანი იყო, რადგან პროექტში რამდენიმე განსხვავებული ტიპის მოდელი გამოიყენება: tree-based მოდელები, classical time-series მოდელები და deep learning მოდელები.

Pipeline-ის მიზანი იყო raw Kaggle ფაილებიდან მიგვეღო ერთიანი, სუფთა და დროით სწორად დალაგებული dataset.

---

### Raw Data Loading

თავდაპირველად იტვირთება ოთხი ფაილი:

- `train.csv.zip`
- `test.csv.zip`
- `features.csv.zip`
- `stores.csv`

შემდეგ `Date` სვეტი გადადის `datetime` ფორმატში, რადგან ყველა time-series ოპერაცია — sorting, splitting, calendar features და weekly grid — თარიღებზეა დამოკიდებული.

---

### მონაცემების გაერთიანება

`train` და `test` მონაცემები ერთიანდება `features.csv` და `stores.csv` ფაილებთან.

`features.csv` ამატებს შემდეგ ცვლადებს:

- Temperature
- Fuel_Price
- MarkDown1–MarkDown5
- CPI
- Unemployment
- IsHoliday

`stores.csv` ამატებს:

- Type
- Size

ამის შემდეგ თითოეული row შეიცავს არა მხოლოდ გაყიდვების ინფორმაციას, არამედ store-level და external economic/promotional features-საც.

---

### Time-based Validation Split

Time-series ამოცანაში random split არასწორია, რადგან მოდელმა მომავლის ინფორმაცია არ უნდა ნახოს. ამიტომ validation set შეიქმნა დროის მიხედვით:

- `train_part` — ძველი პერიოდი
- `valid_part` — ბოლო 3 თვე

Validation split სრულდება `fill_grid()`-მდე, რათა თავიდან ავირიდოთ future Store-Dept existence leakage.

საწყის preprocessing ვერსიაში `fill_grid()` სრულდებოდა split-მდე. ამან შეიძლება შექმნას სუსტი leakage, რადგან validation პერიოდში პირველად გამოჩენილი Store-Dept წყვილები train grid-შიც ხელოვნურად ჩნდებოდა. საბოლოო pipeline-ში ეს გამოსწორდა:

```text
merge raw data
→ time split
→ fill_grid only on train_part

# ML-Final-Walmart-Recruiting-Store-Sales-Forecasting
Machine Learning Final Project

# N-BEATS ექსპერიმენტები

## მოდელის მოკლე აღწერა

N-BEATS (Neural Basis Expansion Analysis for Time Series Forecasting) არის Deep Learning-ზე დაფუძნებული დროითი რიგების პროგნოზირების მოდელი, რომელიც სპეციალურად Forecasting ამოცანებისთვის შეიქმნა. განსხვავებით LSTM-ისა და Transformer-ის მსგავსი მოდელებისგან, N-BEATS არ იყენებს რეკურენტულ ან attention მექანიზმებს. მოდელი აგებულია სრულად Fully Connected (MLP) ბლოკებისგან.

N-BEATS შედგება რამდენიმე Block-ისგან. თითოეული Block სწავლობს დროითი რიგის გარკვეულ ნაწილს (Trend, Seasonality ან Generic Pattern), ხოლო შემდეგ Backcast მექანიზმით ხსნის უკვე ახსნილ ინფორმაციას და Forecast ნაწილს გადასცემს შემდეგ Block-ს. საბოლოო პროგნოზი მიიღება ყველა Block-ის Forecast-ების ჯამით.

ამ პროექტში გამოყენებულია NeuralForecast ბიბლიოთეკის N-BEATS იმპლემენტაცია.

---

# გამოყენებული მონაცემები

მოდელი მუშაობდა **Global Univariate Forecasting** რეჟიმში.

ეს ნიშნავს, რომ თითოეული `(Store, Dept)` წყვილი განიხილებოდა როგორც დამოუკიდებელი დროითი რიგი, თუმცა ერთი გლობალური N-BEATS მოდელი სწავლობდა ყველა სერიაზე ერთდროულად.

მოდელის შესაყვანი DataFrame შედგებოდა მხოლოდ სამი სვეტისგან:

- `unique_id` — Store-Dept იდენტიფიკატორი
- `ds` — თარიღი
- `y` — Weekly Sales

ამ ვერსიაში N-BEATS არ იყენებდა დამატებით Feature-ებს, როგორიცაა:

- Temperature
- Fuel Price
- CPI
- Unemployment
- Store Type
- Store Size
- MarkDown
- Calendar Features

ეს Feature-ები საერთო preprocessing pipeline-ში იქმნებოდა, თუმცა N-BEATS-ის პირველი ვერსია მხოლოდ გაყიდვების ისტორიულ მნიშვნელობებზე იყო დაფუძნებული.

---

# გამოყენებული პარამეტრები

საბოლოოდ საუკეთესო შედეგი მიღებული იქნა შემდეგი პარამეტრებით:

```python
input_size = 104
horizon = 14

learning_rate = 0.0005
batch_size = 32
max_steps = 2000

loss = MAE

scaler_type = "identity"

stack_types = ["identity", "trend", "seasonality"]
n_blocks = [1,1,1]
mlp_units = [[512,512],[512,512],[512,512]]

missing_target_strategy = "linear_interpolation"
```

---

# ჩატარებული ექსპერიმენტები

## 1. საწყისი მოდელი

პირველი ექსპერიმენტი შესრულდა თითქმის Default პარამეტრებით.

გამოყენებული იყო:

- learning_rate = 0.001
- batch_size = 32
- max_steps = 500
- input_size = 104

ამან მოგვცა კარგი საწყისი შედეგი და გამოყენებული იქნა როგორც Reference.

---

## 2. Training Steps

შემოწმდა სხვადასხვა რაოდენობის Training Steps.

გატესტილი მნიშვნელობები:

- 500
- 1000
- 1500
- 2000

აღმოჩნდა, რომ 500 Step არასაკმარისი იყო.

1000 და 1500 მნიშვნელოვნად აუმჯობესებდა შედეგს.

საბოლოოდ საუკეთესო შედეგი მიღებული იქნა 2000 Step-ზე.

---

## 3. Learning Rate

გატესტილი იყო ორი მნიშვნელობა:

- 0.001
- 0.0005

Learning Rate-ის შემცირებამ მნიშვნელოვნად გააუმჯობესა Validation WMAE.

საბოლოოდ არჩეული იქნა:

```
learning_rate = 0.0005
```

---

## 4. Batch Size

გატესტილი იყო:

- 32
- 63

Batch Size-ის გაზრდამ Validation WMAE გააუარესა.

საბოლოოდ დარჩა:

```
batch_size = 32
```

---

## 5. Scaler

NeuralForecast-ში შესაძლებელია Time Series-ის სკალირება Training-მდე.

გატესტილი იყო:

```
identity
```

და

```
robust
```

Robust Scaler-მა გააუარესა როგორც Holiday, ასევე Overall WMAE.

ამიტომ საბოლოოდ გამოყენებული იქნა

```
identity
```

ანუ მონაცემები დამატებითი სკალირების გარეშე.

---

## 6. Target Column

გატესტილი იყო ორი Target:

- Weekly_Sales
- Weekly_Sales_clipped

Weekly_Sales_clipped უარყოფით გაყიდვებს 0-ზე აჭრიდა.

ორივე Target თითქმის იდენტურ შედეგს იძლეოდა.

Raw Weekly_Sales დაახლოებით 1 WMAE-ით უკეთესი აღმოჩნდა, თუმცა განსხვავება იმდენად მცირე იყო, რომ პრაქტიკულად თანაბარ შედეგად ჩაითვალა.

---

## 7. Input Size

გატესტილი იყო:

- 104
- 112

112 კვირიანი ისტორიის გამოყენებამ შედეგი გააუარესა.

საბოლოოდ დარჩა:

```
input_size = 104
```

---

## 8. Model Architecture

გატესტილი იყო უფრო დიდი Fully Connected Network.

Default:

```
mlp_units =
[[512,512],
 [512,512],
 [512,512]]
```

შემდეგ გაიზარდა:

```
[[1024,1024],
 [1024,1024],
 [1024,1024]]
```

ასევე შემოწმდა

```
n_blocks=[2,2,2]
```

ორივე შემთხვევაში Validation WMAE გაუარესდა.

ეს მიუთითებს, რომ მოცემული Dataset-ისთვის Default Architecture საკმარისი აღმოჩნდა.

---

# აღმოჩენილი პრობლემა

ექსპერიმენტების მიმდინარეობისას აღმოჩნდა მნიშვნელოვანი პრობლემა საერთო preprocessing pipeline-ში.

თავდაპირველად `fill_grid()` სრულდებოდა Validation Split-მდე.

```
merge
↓

fill_grid
↓

time_split
```

ამან შეიძლება გამოეწვია Future Store-Dept არსებობის შესახებ ინფორმაციის გაჟონვა (Future Store-Department Existence Leakage), რადგან Grid უკვე შეიცავდა იმ Series-ებსაც, რომლებიც მხოლოდ Validation პერიოდში ჩნდებოდა.

Pipeline გადაიწერა შემდეგნაირად:

```
merge
↓

time_split
↓

fill_grid(train only)
```

Leakage-safe preprocessing-ის დანერგვის შემდეგ, საბოლოო ექსპერიმენტები და საუკეთესო კონფიგურაციები თავიდან გაეშვა. საბოლოო შედეგები და მოდელების შედარება ეფუძნება მხოლოდ ამ კორექტირებულ ექსპერიმენტებს.
---

# საბოლოო შედეგი

საუკეთესო N-BEATS მოდელმა მიიღო:

| Metric | მნიშვნელობა |
|---------|------------:|
| Validation WMAE | **1276.83** |
| Baseline WMAE | **2244.99** |
| Improvement | **43.1%** |

N-BEATS-მა მნიშვნელოვნად აჯობა Store-Department Median Baseline-ს და აჩვენა, რომ მხოლოდ ისტორიული გაყიდვების გამოყენებითაც შესაძლებელია საკმაოდ ზუსტი პროგნოზის მიღება.

---

# დასკვნა

ჩატარებული ექსპერიმენტებიდან გამოიკვეთა, რომ ყველაზე მნიშვნელოვანი გავლენა შედეგზე ჰქონდა:

- Learning Rate-ის შემცირებას;
- Training Steps-ის გაზრდას;
- Input Window-ის სწორ არჩევას.

მეორეს მხრივ, Model-ის ზომის გაზრდამ, დამატებითმა Scaling-მა და Batch Size-ის გაზრდამ შედეგი ვერ გააუმჯობესა.

საბოლოოდ არჩეული იქნა შედარებით მარტივი N-BEATS არქიტექტურა, რომელმაც საუკეთესო შედეგი აჩვენა მოცემულ ამოცანაზე.


## ექსპერიმენტების შეჯამება

| ექსპერიმენტი | ძირითადი ცვლილება | Validation WMAE | შედეგი |
|--------------|-------------------|----------------:|---------|
| Baseline | Store-Dept Median | **2244.99** | Reference |
| N-BEATS v1 | input=104, lr=0.001, steps=500 | ~1300 | საწყისი მოდელი |
| Learning Rate | 0.001 → **0.0005** | გაუმჯობესდა | ✅ დარჩა |
| Training Steps | 500 → 1000 | გაუმჯობესდა | ✅ |
| Training Steps | 1000 → 1500 | გაუმჯობესდა | ✅ |
| Training Steps | 1500 → **2000** | **1277.79** | ✅ საუკეთესო clipped შედეგი |
| Batch Size | 32 → 63 | გაუარესდა | ❌ |
| Scaler | Robust | **1179 → 1295** (Leakage-safe pipeline-ზე ასევე გაუარესდა) | ❌ |
| Target | Weekly_Sales_clipped → Weekly_Sales | **1276.83** | მცირედით უკეთესი |
| Input Window | 104 → 112 | **1282.99** | ❌ |
| Architecture | n_blocks=[2,2,2] | **1312.47** | ❌ |
| Architecture | MLP 1024×1024 | **1294.83** | ❌ |


## საბოლოოდ არჩეული მოდელი

| პარამეტრი | მნიშვნელობა |
|-----------|-------------|
| Model | N-BEATS |
| Forecast Horizon | 14 კვირა |
| Input Window | 104 კვირა |
| Learning Rate | 0.0005 |
| Batch Size | 32 |
| Training Steps | 2000 |
| Loss | MAE |
| Target | Weekly_Sales *(ან Weekly_Sales_clipped, თუ მას დატოვებ)* |
| Missing Target Strategy | Linear Interpolation |
| Scaler | Identity |
| Stack Types | Identity + Trend + Seasonality |
| Blocks | [1,1,1] |
| MLP Units | [[512,512],[512,512],[512,512]] |
| Validation WMAE | **1276.83** *(ან 1277.79 clipped-ის შემთხვევაში)* |
| Baseline WMAE | **2244.99** |
| Improvement over Baseline | **43.1%** |
