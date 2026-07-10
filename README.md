### მნიშვნელოვანი დაკვირვება: დროითი სერიების უწყვეტობა

TFT მოდელის იმპლემენტაციის დროს აღმოვაჩინეთ მონაცემების მნიშვნელოვანი სტრუქტურული თავისებურება, რომელიც წინა ექსპერიმენტებში (N-BEATS და DLinear) არ გამოჩენილა.

EDA-ის გაფართოების შემდეგ აღმოჩნდა, რომ `Store + Dept` დროითი სერიები სრულად უწყვეტი არ არის. Time-based split-ის შემდეგ:

- Training ნაწილში იყო **3,321** უნიკალური `Store + Dept` სერია;
- Validation ნაწილში იყო **3,104** სერია;
- **227** სერია არსებობდა მხოლოდ training პერიოდში და validation-ში აღარ გვხვდებოდა;
- **10** ახალი სერია პირველად მხოლოდ validation პერიოდში გამოჩნდა.

გარდა ამისა, validation პერიოდი 14 კვირას მოიცავდა, თუმცა მხოლოდ **2,803** სერიას ჰქონდა ყველა 14 კვირის ჩანაწერი. დანარჩენ სერიებს validation-ში მხოლოდ 1-13 კვირა ჰქონდათ.

N-BEATS და DLinear-ის შემთხვევაში ეს პრობლემა არ გამოვლინდა, რადგან ორივე მოდელი პროგნოზს ქმნის მხოლოდ training მონაცემებზე დაყრდნობით და ავტომატურად აგენერირებს ყველა ცნობილი სერიისთვის მომდევნო `h=14` კვირის პროგნოზს. შეფასების ეტაპზე ეს პროგნოზები უბრალოდ ებმება რეალურად არსებულ validation მონაცემებს, ხოლო ის პროგნოზები, რომლებსაც შესაბამისი რეალური observation არ გააჩნიათ, WMAE-ის გამოთვლაში არ მონაწილეობს.

TFT განსხვავებულად მუშაობს. რადგან მოდელი იყენებს **future covariates**-ს (მაგალითად `Temperature`, `Fuel_Price`, `IsHoliday`, `MarkDown`, `CPI`, `Unemployment` და კალენდარულ feature-ებს), პროგნოზის გაკეთებისას მას სჭირდება სრული future dataframe ყველა training სერიისა და ყველა მომდევნო 14 კვირისთვის. Validation dataframe-ში ასეთი სრული ბადე არ არსებობდა, რის გამოც NeuralForecast აბრუნებდა შეცდომას:

> `There are missing combinations of ids and times in futr_df.`

ამ პრობლემის გადასაჭრელად საჭიროა prediction-ისთვის ცალკე სრული future dataframe-ის აგება. იგი უნდა შეიქმნას training-ში არსებული ყველა `unique_id`-ისთვის და ყველა მომდევნო 14 კვირისთვის (`make_future_dataframe()`), შემდეგ კი მას უნდა დაემატოს შესაბამისი future covariates. შეფასება კვლავ მხოლოდ რეალურად არსებულ validation observations-ზე ხდება, ამიტომ მოდელის შეფასების პროცესი უცვლელი რჩება.

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

# N-BEATS

## მოდელის ზოგადი აღწერა

N-BEATS (Neural Basis Expansion Analysis for Time Series Forecasting) არის Deep Learning-ზე დაფუძნებული დროითი რიგების პროგნოზირების არქიტექტურა. მისი მთავარი იდეაა, რომ პროგნოზი აშენდეს მხოლოდ წარსული მნიშვნელობების საფუძველზე, რთული ხელით შექმნილი Feature Engineering-ის გარეშე. განსხვავებით RNN/LSTM ტიპის მოდელებისგან, N-BEATS არ იყენებს რეკურენტულ სტრუქტურას, ხოლო Transformer-ებისგან განსხვავებით არ არის აგებული Attention მექანიზმზე. მოდელის ძირითადი ნაწილი არის Fully Connected / MLP ბლოკები.

ამ პროექტში N-BEATS გამოვიყენეთ როგორც **Global Univariate Forecasting** მოდელი. Walmart-ის ამოცანაში თითოეული `(Store, Dept)` წყვილი დამოუკიდებელ დროით რიგს წარმოადგენს. მაგალითად, `Store=1, Dept=1` არის ერთი სერია, `Store=1, Dept=2` — მეორე და ასე შემდეგ. თუმცა ცალკე მოდელის დატრენინგების ნაცვლად, ერთი გლობალური N-BEATS მოდელი სწავლობდა ყველა Store-Department სერიაზე ერთად. ეს მიდგომა სასარგებლოა, რადგან მოდელს შეუძლია სხვადასხვა დეპარტამენტებსა და მაღაზიებში არსებული საერთო სეზონური ან გაყიდვების მსგავსი pattern-ები გამოიყენოს.

მოდელის input იყო მხოლოდ:

- `unique_id` — Store-Department იდენტიფიკატორი;
- `ds` — თარიღი;
- `y` — გაყიდვების target მნიშვნელობა.

ამ N-BEATS ექსპერიმენტებში მოდელი **არ იყენებდა** დამატებით ცვლადებს, როგორიცაა MarkDown, Temperature, Fuel Price, CPI, Unemployment, Store Type ან Store Size. ეს Feature-ები preprocessing pipeline-ში მაინც მზადდებოდა, რადგან ისინი საჭირო იყო სხვა მოდელებისთვის, განსაკუთრებით LightGBM/XGBoost-ისთვის და მომავალში covariate-based Deep Learning მოდელებისთვის. N-BEATS-ის ამ ვერსიის მიზანი იყო შეგვეფასებინა, რამდენად კარგად მუშაობს pure historical-sales forecasting მიდგომა.

---

## N-BEATS არქიტექტურა

N-BEATS შედგება რამდენიმე თანმიმდევრული Block-ისგან. თითოეული Block იღებს წარსული გაყიდვების ფანჯარას და სწავლობს ორი ტიპის output-ს:

- **Backcast** — ისტორიული input-ის ახსნა;
- **Forecast** — მომავალი პერიოდის პროგნოზი.

Backcast-ის მიზანია უკვე ნანახი სიგნალის გარკვეული ნაწილის ახსნა. შემდეგ ეს ახსნილი ნაწილი აკლდება input-ს და დარჩენილი residual გადაეცემა შემდეგ Block-ს. ამ გზით ყოველი შემდეგი Block სწავლობს იმას, რაც წინა Block-ებმა ვერ ახსნეს. საბოლოო პროგნოზი მიიღება ყველა Block-ის Forecast კომპონენტების ჯამით.

სქემატურად:

```text
Input history
   ↓
Block 1 → backcast_1 + forecast_1
   ↓
Residual after Block 1
   ↓
Block 2 → backcast_2 + forecast_2
   ↓
Residual after Block 2
   ↓
...
Final forecast = forecast_1 + forecast_2 + ...
```

N-BEATS-ში Block-ები ერთიანდება Stack-ებად. NeuralForecast-ის იმპლემენტაციაში მნიშვნელოვანია რამდენიმე Stack Type:

- **Identity** — generic block, რომელიც სწავლობს წინასწარ არამკაცრად განსაზღვრულ pattern-ებს;
- **Trend** — სწავლობს გრძელვადიან ტრენდს polynomial basis-ის გამოყენებით;
- **Seasonality** — სწავლობს პერიოდულობას harmonic/Fourier basis-ის საშუალებით.

საწყისად გამოყენებული იყო NeuralForecast-ის default ტიპის არქიტექტურა: `Identity + Trend + Seasonality`. მოგვიანებით ჩავატარეთ არქიტექტურული ექსპერიმენტები და აღმოჩნდა, რომ Walmart-ის ამ მონაცემებზე უკეთ იმუშავა უფრო სპეციალიზებულმა `Trend + Seasonality` არქიტექტურამ, სადაც გაიზარდა Trend polynomial degree, Harmonic basis და MLP layer-ების ზომა.

---

## Preprocessing და Leakage-safe Pipeline

N-BEATS ექსპერიმენტები ეყრდნობოდა საერთო preprocessing pipeline-ს. Pipeline-ის მიზანი იყო raw Kaggle ფაილებიდან სუფთა, დროით სწორად დალაგებული dataset-ის მიღება.

იტვირთებოდა ოთხი ძირითადი ფაილი:

- `train.csv.zip`;
- `test.csv.zip`;
- `features.csv.zip`;
- `stores.csv`.

შემდეგ `Date` სვეტი გადადიოდა `datetime` ფორმატში, რადგან time-series split, weekly grid და calendar features სწორედ თარიღებზეა დამოკიდებული.

`train` და `test` მონაცემები ერთიანდებოდა `features.csv` და `stores.csv` ფაილებთან. შედეგად თითოეულ row-ს ემატებოდა external და store-level ინფორმაცია:

- Temperature;
- Fuel_Price;
- MarkDown1–MarkDown5;
- CPI;
- Unemployment;
- IsHoliday;
- Type;
- Size.

### Time-based validation split

Time-series forecasting-ში random split არასწორია, რადგან მომავალის ინფორმაცია არ უნდა მოხვდეს train-ში. ამიტომ validation set შეიქმნა დროით:

- `train_part` — ძველი პერიოდი;
- `valid_part` — ბოლო 3 თვე.

პირველ preprocessing ვერსიაში `fill_grid()` სრულდებოდა split-მდე:

```text
merge raw data
→ fill_grid
→ time_split
```

ამან შექმნა სუსტი leakage-ის რისკი. პრობლემა ის იყო, რომ თუ რომელიმე Store-Department წყვილი პირველად მხოლოდ validation პერიოდში ჩნდებოდა, `fill_grid()` მას full train range-ის basis-ზე უკვე training grid-შიც ხელოვნურად ჩასვამდა. ეს პირდაპირ target leakage არ იყო, რადგან validation-ის გაყიდვები train-ში არ ხვდებოდა, მაგრამ მოდელი იგებდა future Store-Department pair-ის არსებობას.

შემდეგ pipeline შეიცვალა:

```text
merge raw data
→ time_split
→ fill_grid only on train_part
```

ამ ცვლილების შემდეგ training grid იქმნებოდა მხოლოდ იმ Store-Department წყვილებისთვის, რომლებიც რეალურად train პერიოდში არსებობდა. საბოლოო შედეგები ეფუძნება სწორედ ამ **leakage-safe preprocessing** ვერსიას.

### `fill_grid()`

Walmart-ის მონაცემებში თითოეული Store-Department წყვილი weekly time series-ია. ზოგჯერ რომელიმე კვირა საერთოდ არ გვხვდება raw data-ში. `fill_grid()` ყველა train-period Store-Department წყვილისთვის ქმნის ყველა პარასკევის row-ს.

მაგალითად, თუ გვაქვს:

| Store | Dept | Date | Weekly_Sales |
|---:|---:|---|---:|
| 1 | 1 | 2010-02-05 | 20000 |
| 1 | 1 | 2010-02-12 | 22000 |
| 1 | 1 | 2010-02-26 | 21000 |

აკლია `2010-02-19`. `fill_grid()` ამატებს ამ row-ს:

| Store | Dept | Date | Weekly_Sales |
|---:|---:|---|---:|
| 1 | 1 | 2010-02-05 | 20000 |
| 1 | 1 | 2010-02-12 | 22000 |
| 1 | 1 | 2010-02-19 | NaN |
| 1 | 1 | 2010-02-26 | 21000 |

მნიშვნელოვანია, რომ `fill_grid()` არ ავსებს target-ს 0-ით. ის მხოლოდ ამატებს row-ს და `Weekly_Sales` რჩება `NaN`. N-BEATS-ის model-specific preprocessing-ში შემდეგ გატესტილი იყო missing target values-ის შევსების ორი მეთოდი:

- `zero`;
- `linear_interpolation`.

საბოლოოდ უკეთ იმუშავა `linear_interpolation`-მა. ეს ნიშნავს, რომ grid-filled missing weeks განვიხილეთ როგორც missing observations და არა აუცილებლად true zero-sales კვირები.

### Negative sales

Raw `Weekly_Sales` ზოგჯერ უარყოფითია. ეს შეიძლება გამოწვეული იყოს returns/refunds/adjustments-ით. preprocessing-ში შეიქმნა:

- `is_negative_sales`;
- `Weekly_Sales_clipped`.

`Weekly_Sales_clipped` არის:

```text
Weekly_Sales_clipped = max(Weekly_Sales, 0)
```

N-BEATS-ზე გატესტილი იყო როგორც raw `Weekly_Sales`, ასევე `Weekly_Sales_clipped`. საბოლოო საუკეთესო კონფიგურაციაში გამოყენებული იქნა `Weekly_Sales_clipped`, რადგან საუკეთესო არქიტექტურულმა მოდელმა სწორედ ამ target-ით აჩვენა საუკეთესო შედეგი.

---

# ექსპერიმენტების სტრატეგია

N-BEATS ექსპერიმენტები ჩატარდა ეტაპობრივად. მიზანი იყო არა მხოლოდ საუკეთესო score-ის მიღება, არამედ იმის გაგება, რომელი პარამეტრი რა გავლენას ახდენდა მოდელზე.

ტესტირება დაიწყო baseline configuration-ით, შემდეგ ცალ-ცალკე შეიცვალა:

- training steps;
- learning rate;
- batch size;
- scaler;
- target definition;
- input window;
- model capacity;
- number of blocks;
- stack architecture;
- trend/seasonality basis.

ყველა მნიშვნელოვანი run დაილოგა MLflow-ში. საბოლოო შედეგების შედარება ეფუძნება leakage-safe preprocessing pipeline-ზე მიღებულ შედეგებს.

---

## 1. საწყისი მოდელი

პირველი N-BEATS მოდელი გაეშვა თითქმის default პარამეტრებით:

```text
input_size = 104
learning_rate = 0.001
batch_size = 32
max_steps = 500
target_transform = none
```

`input_size=104` შეირჩა იმიტომ, რომ 104 კვირა დაახლოებით ორ სრულ წლიურ სეზონურ ციკლს მოიცავს. Walmart-ის გაყიდვებში სეზონურობა ძალიან მნიშვნელოვანია, განსაკუთრებით holiday weeks-ის გამო, ამიტომ ერთი წლის ისტორია არასაკმარისი შეიძლება ყოფილიყო.

საწყისი მოდელი უკვე მნიშვნელოვნად სჯობდა Store-Department median baseline-ს, მაგრამ training steps-ის გაზრდამ შემდგომში აჩვენა, რომ 500 step არასაკმარისი იყო.

---

## 2. Training Steps

`max_steps` განსაზღვრავს რამდენი optimization step შესრულდება training-ის დროს. ძალიან მცირე მნიშვნელობა იწვევს underfitting-ს, რადგან მოდელი ვერ ასწრებს pattern-ების სწავლას. ზედმეტად დიდი მნიშვნელობა შეიძლება overfitting-ს იწვევდეს.

გატესტილი იყო:

- 500;
- 1000;
- 1500;
- 2000.

ძველ pipeline-ზე 500 → 1000 → 1500 მკაფიოდ აუმჯობესებდა შედეგს. leakage-safe pipeline-ზე საბოლოო საუკეთესო კონფიგურაციებში 2000 step-მა უკეთ იმუშავა. ეს მიუთითებს, რომ სწორად გაწმენდილ training setup-ში მოდელს მეტი optimization step სჭირდებოდა trend/seasonality კომპონენტების უკეთ დასასწავლად.

საბოლოოდ გამოყენებული იქნა:

```text
max_steps = 2000
```

---

## 3. Learning Rate

`learning_rate` განსაზღვრავს, რამდენად დიდი ნაბიჯით იცვლება მოდელის წონები ყოველ update-ზე.

გატესტილი იყო:

- `0.001`;
- `0.0005`.

Learning rate-ის შემცირებამ ერთ-ერთი ყველაზე მნიშვნელოვანი გაუმჯობესება მოიტანა. `0.001` ზედმეტად აგრესიული აღმოჩნდა: მოდელი სწავლობდა სწრაფად, მაგრამ უფრო ნაკლებად სტაბილურად. `0.0005`-ზე optimization უფრო რბილი გახდა და validation WMAE შემცირდა.

საბოლოოდ გამოყენებული იქნა:

```text
learning_rate = 0.0005
```

---

## 4. Batch Size

`batch_size` განსაზღვრავს რამდენი training window მუშავდება ერთ gradient update-ზე.

გატესტილი იყო:

- `32`;
- `63/64`.

Batch size-ის გაზრდამ overall WMAE გააუარესა. დიდი batch იძლევა უფრო სტაბილურ gradient-ს, მაგრამ ხშირად ამცირებს gradient noise-ს, რომელიც ზოგჯერ უკეთეს generalization-ს ეხმარება. ამ dataset-ზე მცირე batch უკეთესი აღმოჩნდა.

საბოლოოდ დარჩა:

```text
batch_size = 32
```

---

## 5. Missing Target Strategy

`fill_grid()`-ის შემდეგ ზოგიერთი row ხელოვნურად არის დამატებული და target არის missing. ამ row-ებისთვის გატესტილი იყო:

- zero filling;
- linear interpolation.

Zero filling გულისხმობს, რომ missing week განვიხილოთ როგორც 0 გაყიდვა. Linear interpolation missing value-ს ავსებს იმავე Store-Department სერიის მეზობელი წერტილების მიხედვით.

Linear interpolation-მა უკეთ იმუშავა. ეს მიუთითებს, რომ grid-filled missing weeks უფრო ხშირად ჰგავდა missing observations-ს და არა აუცილებლად real zero-sales week-ს. N-BEATS როგორც univariate sequence model ძალიან მგრძნობიარეა target sequence-ის ფორმაზე, ამიტომ ხელოვნურმა ნულებმა შეიძლება ზედმეტი sharp drops შეიტანოს სერიაში. Interpolation უფრო გლუვ და რეალისტურ sequence-ს ქმნიდა.

საბოლოოდ გამოყენებული იქნა:

```text
missing_target_strategy = linear_interpolation
```

---

## 6. Scaler

NeuralForecast იძლევა scaler-ის არჩევის საშუალებას. გატესტილი იყო:

- identity;
- robust.

`identity` ნიშნავს, რომ დამატებითი scaling არ გამოიყენება. `robust` ცდილობს extreme values-ის გავლენის შემცირებას.

Robust scaler-მა გააუარესა შედეგი, განსაკუთრებით holiday weeks-ზე. Walmart data-ში holiday weeks ხშირად არის სწორედ ის პერიოდები, სადაც გაყიდვები მკვეთრად იზრდება. Robust scaling ამ მაღალი მნიშვნელობების გავლენას ამცირებს და მოდელს შეიძლება გაუჭირდეს holiday spikes-ის სწორად სწავლა.

საბოლოოდ გამოყენებული იქნა:

```text
scaler_type = identity
```

---

## 7. Target Column

გატესტილი იყო ორი target:

- `Weekly_Sales`;
- `Weekly_Sales_clipped`.

Raw `Weekly_Sales` ზოგიერთ default-architecture run-ში ძალიან მცირედით უკეთესი იყო, მაგრამ სხვაობა დაახლოებით 1 WMAE-ის ფარგლებში იყო. საბოლოო საუკეთესო არქიტექტურულ configuration-ში საუკეთესო შედეგი მივიღეთ `Weekly_Sales_clipped` target-ით.

`Weekly_Sales_clipped` ასევე უფრო კონსერვატიულია, რადგან უარყოფით გაყიდვებს 0-მდე ჭრის და forecast-ის ამოცანაში უარყოფითი გაყიდვების სწავლა ნაკლებად სასურველია.

საბოლოოდ საუკეთესო მოდელში გამოყენებული იქნა:

```text
target_col = Weekly_Sales_clipped
```

---

## 8. Input Window

`input_size` განსაზღვრავს, რამდენი წარსული კვირა მიეწოდება მოდელს პროგნოზის გასაკეთებლად.

გატესტილი იყო:

- 104;
- 112.

104 კვირა მოიცავს დაახლოებით ორ წლიურ სეზონურ ციკლს. 112 კვირამ validation WMAE გააუარესა. დამატებითი ისტორია უკვე აღარ მატებდა ბევრ ახალ სეზონურ ინფორმაციას, მაგრამ ზრდიდა sequence-ის სირთულეს და noise-ს.

საბოლოოდ დარჩა:

```text
input_size = 104
```

---

## 9. Fully Connected Layer-ების ზომა

N-BEATS-ის block-ები აგებულია MLP layer-ებით. თავდაპირველად გამოიყენებოდა default:

```text
[[512,512],
 [512,512],
 [512,512]]
```

შემდეგ შემოწმდა უფრო დიდი layer-ები:

```text
[[1024,1024],
 [1024,1024]]
```

და საბოლოო არქიტექტურულ ექსპერიმენტში:

```text
[[1024,1024],
 [1024,1024],
 [1024,1024]]
```

მხოლოდ MLP-ის ზომის ზრდა default architecture-ზე არ იყო საკმარისი და ზოგიერთ run-ში შედეგს აუარესებდა. თუმცა Trend + Seasonality architecture-თან ერთად 1024-იანი MLP layers-მა საუკეთესო შედეგი მოგვცა. ეს აჩვენებს, რომ capacity-ის გაზრდა თავისთავად არ არის საკმარისი; ის ეფექტური გახდა მხოლოდ მაშინ, როცა model structure უკეთ მოერგო გაყიდვების trend/seasonality ბუნებას.

---

## 10. Block-ების რაოდენობა

გატესტილი იყო:

```text
n_blocks = [2,2,2]
```

default-ის ნაცვლად:

```text
n_blocks = [1,1,1]
```

Block-ების რაოდენობის გაზრდამ validation WMAE გააუარესა. ეს ნიშნავს, რომ მეტი residual decomposition ამ მონაცემისთვის ზედმეტი აღმოჩნდა. დამატებითმა block-ებმა გაზარდა მოდელის სირთულე, მაგრამ არ გააუმჯობესა generalization.

---

## 11. Trend / Seasonality Architecture

ბოლო ეტაპზე დეტალურად შემოწმდა N-BEATS-ის შიდა არქიტექტურა. შეიცვალა:

- Stack Types;
- Trend Polynomial Degree;
- Harmonic Basis;
- Seasonality Stack-ის სიღრმე;
- Fully Connected Layer-ების ზომა.

Architecture sweep-ის შედეგები:

| Architecture | Validation WMAE | Holiday WMAE | Non-Holiday WMAE |
|---|---:|---:|---:|
| Trend + Seasonality | 1289.64 | 1402.17 | 1246.29 |
| Trend + Seasonality + Harmonics = 5 | 1276.95 | 1385.33 | 1235.21 |
| Trend Degree = 3 | **1276.74** | 1398.02 | 1230.02 |
| Deeper Seasonality | 1286.70 | 1356.11 | 1259.96 |
| Identity + Trend + Seasonality | 1294.83 | 1447.48 | 1236.02 |

თავდაპირველად გამოყენებული იყო default `Identity + Trend + Seasonality` არქიტექტურა. Sweep-მა აჩვენა, რომ Walmart-ის მონაცემებისთვის უკეთესი აღმოჩნდა უფრო ფოკუსირებული `Trend + Seasonality` სტრუქტურა.

საუკეთესო შედეგი მიღებული იქნა შემდეგი კონფიგურაციით:

```text
Stack Types       = Trend + Seasonality
Blocks            = [1,1]
Polynomial Degree = 3
Harmonics         = 3
MLP Units         = [[1024,1024],
                     [1024,1024],
                     [1024,1024]]
```

Trend Stack-ის polynomial degree-ის გაზრდამ მოდელს მისცა საშუალება უფრო მოქნილი გრძელვადიანი trend ესწავლა. Harmonic basis-ის გამოყენებამ seasonality component უფრო მდიდარი გახადა. Fully Connected layer-ების გაფართოებამ კი გაზარდა model capacity, მაგრამ უკვე სწორად შერჩეულ trend/seasonality structure-ში.

ერთ მომენტში იგივე არქიტექტურის დამოუკიდებელ run-ზე შედეგი გაუარესდა. შემოწმების შემდეგ აღმოჩნდა, რომ run ზუსტად იგივე configuration-ით არ იყო გაშვებული: `mlp_units` განსხვავებული იყო. სწორი configuration-ის ხელახლა გაშვებისას იგივე საუკეთესო შედეგი განმეორდა:

```text
Validation WMAE = 1276.74
```

ამიტომ საბოლოოდ დადგინდა, რომ გაუმჯობესება შემთხვევითი არ ყოფილა. საუკეთესო შედეგი მოგვცა არა default architecture-მა, არამედ trend/seasonality-ზე მორგებულმა architecture-მა.

---

# Leakage-safe შედეგები და Baseline

საბოლოო შეფასება ეფუძნება leakage-safe preprocessing pipeline-ს.

Baseline იყო Store-Department median baseline:

```text
Baseline WMAE = 2244.99
```

საუკეთესო N-BEATS შედეგი:

```text
Validation WMAE = 1276.74
```

გაუმჯობესება:

```text
Improvement = 968.25 WMAE
Improvement % = 43.13%
```

---

# ექსპერიმენტების შეჯამება

| ექსპერიმენტი | ცვლილება | შედეგი | ინტერპრეტაცია |
|---|---|---:|---|
| Baseline | Store-Dept Median | 2244.99 | Reference |
| Learning Rate | 0.001 → 0.0005 | გაუმჯობესდა | უფრო სტაბილური optimization |
| Training Steps | 500 → 2000 | გაუმჯობესდა | მოდელს მეტი დრო დასჭირდა pattern-ების სასწავლად |
| Batch Size | 32 → 63/64 | გაუარესდა | მცირე batch უკეთ გენერალიზდებოდა |
| Missing Target | Zero → Linear Interpolation | გაუმჯობესდა | ხელოვნური ნულების ნაცვლად smoother sequence |
| Scaler | Identity → Robust | გაუარესდა | holiday spikes-ის გავლენა შემცირდა |
| Target | Weekly_Sales vs clipped | მსგავსი | clipped საბოლოო არქიტექტურაზე საუკეთესო |
| Input Window | 104 → 112 | გაუარესდა | დამატებითმა ისტორიამ noise გაზარდა |
| Blocks | [1,1,1] → [2,2,2] | გაუარესდა | ზედმეტი model complexity |
| MLP Size | 512 → 1024 | mixed | მხოლოდ trend/seasonality architecture-ში იმუშავა |
| Architecture | Default → Trend + Seasonality | **საუკეთესო** | მონაცემების trend/seasonality ბუნებას უკეთ მოერგო |

---

# საბოლოოდ არჩეული მოდელი

| პარამეტრი | მნიშვნელობა |
|---|---|
| Model | N-BEATS |
| Model Type | Global Univariate |
| Forecast Horizon | 14 კვირა |
| Input Window | 104 კვირა |
| Learning Rate | 0.0005 |
| Batch Size | 32 |
| Training Steps | 2000 |
| Loss | MAE |
| Target | Weekly_Sales_clipped |
| Missing Target Strategy | Linear Interpolation |
| Scaler | Identity |
| Stack Types | Trend + Seasonality |
| Blocks | [1,1] |
| Polynomial Degree | 3 |
| Harmonics | 3 |
| MLP Units | [[1024,1024],[1024,1024],[1024,1024]] |
| Validation WMAE | **1276.74** |
| Baseline WMAE | **2244.99** |
| Improvement | **43.13%** |

---

# Overfitting / Generalization Analysis

საუკეთესო run-ზე train-tail WMAE იყო:

```text
train_tail_wmae = 1806.29
```

ხოლო validation WMAE:

```text
valid_wmae = 1276.74
```

Generalization gap ითვლებოდა ასე:

```text
valid_wmae - train_tail_wmae = -529.55
```

ეს კლასიკურ overfitting-ს არ აჩვენებს, რადგან validation error train-tail error-ზე უარესი არ არის. პირიქით, validation period უფრო მარტივი აღმოჩნდა, ან train-tail split შეიცავდა უფრო რთულ Store-Department-week შემთხვევებს.

ამიტომ overfitting-ის მტკიცებულება ამ შედეგებში არ ჩანს. თუმცა ეს არ ნიშნავს, რომ მოდელი იდეალურად generalizes ყველა პერიოდში. უფრო სწორი ინტერპრეტაციაა, რომ train-tail და validation periods სირთულით განსხვავდებოდა.

---

# დასკვნა

N-BEATS-მა მნიშვნელოვნად აჯობა Store-Department median baseline-ს და აჩვენა, რომ მხოლოდ historical sales sequence-ზეც შესაძლებელია ძლიერი პროგნოზის აგება.

ყველაზე დიდი გავლენა შედეგზე ჰქონდა:

- learning rate-ის შემცირებას;
- training steps-ის გაზრდას;
- missing target values-ის linear interpolation-ით შევსებას;
- N-BEATS-ის architecture-ის trend/seasonality-ზე მორგებას.

მეორეს მხრივ, შედეგი გააუარესა:

- batch size-ის გაზრდამ;
- robust scaler-ის გამოყენებამ;
- input window-ის 112-მდე გაზრდამ;
- block-ების რაოდენობის ზრდამ;
- არასწორ architecture/capacity კომბინაციებმა.

საბოლოო შედეგი აჩვენებს, რომ N-BEATS-ისთვის მხოლოდ ჰიპერპარამეტრების tuning არ იყო საკმარისი. საუკეთესო შედეგი მივიღეთ მაშინ, როცა მოდელის შიდა არქიტექტურა უკეთ მოვარგეთ Walmart sales-ის ბუნებას — ანუ trend-სა და seasonality-ს.




## DLinear

### მოდელის მიმოხილვა

DLinear (Decomposition Linear Model) წარმოადგენს მარტივ, მაგრამ ეფექტურ Deep Learning არქიტექტურას, რომელიც სპეციალურად დროითი რიგების პროგნოზირებისთვის შეიქმნა. მოდელი პირველად წარმოდგენილი იყო LTSF-Linear (Long-Term Series Forecasting with Linear Models) კვლევაში, სადაც ავტორებმა აჩვენეს, რომ გარკვეულ ამოცანებზე მარტივი Linear მოდელები კონკურენციას უწევდნენ გაცილებით რთულ Transformer არქიტექტურებს.

DLinear-ის მთავარი იდეა არის დროითი რიგის **Trend** და **Seasonality** კომპონენტებად გაყოფა (Series Decomposition). ამისათვის გამოიყენება მოძრავი საშუალო (Moving Average), რის შემდეგაც თითოეულ კომპონენტზე დამოუკიდებლად ისწავლება Linear Transformation. საბოლოო პროგნოზი მიიღება ამ ორი კომპონენტის ჯამით.

N-BEATS-ისგან განსხვავებით, DLinear არ იყენებს ღრმა Fully Connected ბლოკებს, Residual Connections-ს ან რთულ არაწრფივ ტრანსფორმაციებს. მისი არქიტექტურა მნიშვნელოვნად მარტივია, რის გამოც ტრენინგი სწრაფია, ხოლო ჰიპერპარამეტრების რაოდენობა შედარებით მცირეა.

ამ პროექტში გამოყენებულია **NeuralForecast** ბიბლიოთეკის DLinear იმპლემენტაცია.

---

## ექსპერიმენტების მიზანი

DLinear მოდელის ექსპერიმენტების მთავარი მიზანი იყო ისეთი ჰიპერპარამეტრების მოძებნა, რომლებიც Walmart-ის გაყიდვების მონაცემებზე საუკეთესო პროგნოზს უზრუნველყოფდა.

ყველა ექსპერიმენტში გამოყენებული იყო ერთი და იგივე:

- Loss Function: **MAE**
- Forecast Horizon: **14 კვირა**
- Validation Split: **ბოლო 3 თვე**
- Evaluation Metric: **Weighted Mean Absolute Error (WMAE)**
- Optimizer: NeuralForecast Default (Adam)
- Random Seed: **42**

ყველა ექსპერიმენტი დალოგილია **MLflow**-ში.

---

## ჩატარებული ექსპერიმენტები

### 1. Input Window (`input_size`)

პირველი ექსპერიმენტები ჩატარდა სხვადასხვა ისტორიული ფანჯრის გამოყენებით.

| Input Size | Validation WMAE |
|-----------:|---------------:|
| 52 | 1544.70 |
| 104 | 1350.43 |
| **112** | **1340.99** ✅ |

**დასკვნა**

52 კვირის ისტორია არასაკმარისი აღმოჩნდა. ისტორიის გაზრდამ მნიშვნელოვნად გააუმჯობესა პროგნოზის ხარისხი, ხოლო საუკეთესო შედეგი მიიღო **112 კვირიანმა Input Window-მ**.

---

### 2. Moving Average Window

DLinear-ის ყველაზე მნიშვნელოვანი მოდელური პარამეტრია **Moving Average Window**, რომელიც გამოიყენება Trend და Seasonality კომპონენტების გასაყოფად.

შემოწმებული იქნა რამდენიმე მნიშვნელობა.

| Moving Average Window | Validation WMAE |
|----------------------:|---------------:|
| 7 | 1343.15 |
| **13** | **1340.99** ✅ |
| 21 | 1342.31 |
| 25 | 1343.86 |
| 51 | 1553.29 |

**დასკვნა**

ძალიან დიდი Moving Average Window ზედმეტად აგლუვებდა დროით რიგს და აუარესებდა შედეგებს, ხოლო ძალიან პატარა მნიშვნელობაც ოპტიმალური არ აღმოჩნდა. საუკეთესო შედეგი მიიღო **13 კვირიანმა Moving Average Window-მ**, რომელმაც საუკეთესო ბალანსი უზრუნველყო Trend-ისა და Seasonality-ის გამოყოფაში.

---

### 3. Learning Rate

Learning Rate-ის გავლენის შესაფასებლად გამოყენებული იქნა რამდენიმე მნიშვნელობა.

| Learning Rate | Validation WMAE |
|--------------:|---------------:|
| 0.0005 | 1379.70 |
| **0.001** | **1340.99** ✅ |

**დასკვნა**

Learning Rate-ის გაზრდამ მნიშვნელოვნად დააჩქარა მოდელის კონვერგენცია და საბოლოო შედეგიც გააუმჯობესა.

---

### 4. Batch Size

შემოწმდა სხვადასხვა Batch Size.

| Batch Size | Validation WMAE |
|-----------:|---------------:|
| 32 | 1553.48 |
| 64 | 1405.42 |
| **128** | **1340.99** ✅ |
| 256 | 1380.49 |

**დასკვნა**

Batch Size-ის ზრდა 32-დან 128-მდე თანდათან აუმჯობესებდა შედეგებს. 256-ზე გადასვლისას გაუმჯობესება აღარ გაგრძელდა, რის გამოც საუკეთესო მნიშვნელობად შეირჩა **128**.

---

### 5. Training Steps

შემოწმდა ტრენინგის ხანგრძლივობაც.

| Max Steps | Validation WMAE |
|----------:|---------------:|
| 1500 | 1553.48 |
| 2000 | 1464.15 |
| **2500** | **1340.99** ✅ |
| 3000 | 1413.35 |

**დასკვნა**

2500 ნაბიჯამდე ტრენინგის გაგრძელება მნიშვნელოვნად აუმჯობესებდა მოდელის ხარისხს, თუმცა 3000 ნაბიჯზე დამატებითი გაუმჯობესება აღარ დაფიქსირდა.

---

### 6. Feature Scaling

ასევე შედარდა სხვადასხვა Scaling Strategy.

| Scaler | Validation WMAE |
|--------|---------------:|
| Identity | 1350.43 |
| **Robust** | **1340.99** ✅ |

**დასკვნა**

Robust Scaler-მა ოდნავ უკეთესი შედეგი აჩვენა, განსაკუთრებით სხვადასხვა Store/Department სერიების განსხვავებული მასშტაბების პირობებში.

---

### 7. Target Variable

დატესტილი იყო ორი განსხვავებული Target.

| Target | Validation WMAE |
|--------|---------------:|
| Weekly_Sales_Clipped | 1350.55 |
| **Weekly_Sales** | **1350.43** |

**დასკვნა**

უარყოფითი გაყიდვების Clipping-მა პრაქტიკულად არ შეცვალა შედეგი. საბოლოოდ არჩეული იქნა **ორიგინალური Weekly_Sales**, რადგან იგი სრულად შეესაბამება Kaggle-ის შეფასების მეტრიკას.

---

## საუკეთესო კონფიგურაცია

| Parameter | Value |
|-----------|------|
| Model | DLinear |
| Target | Weekly_Sales |
| Evaluation Target | Weekly_Sales |
| Input Size | **112** |
| Horizon | 14 |
| Learning Rate | **0.001** |
| Batch Size | **128** |
| Max Steps | **2500** |
| Moving Average Window | **13** |
| Scaler | **Robust** |
| Loss | MAE |

---

## საბოლოო შედეგები

| Metric | Value |
|--------|------:|
| Validation WMAE | **1340.99** |
| Holiday WMAE | **1356.24** |
| Non-Holiday WMAE | **1335.12** |
| Baseline WMAE | **2245.16** |
| Improvement over Baseline | **904.17** |
| Improvement (%) | **40.27%** |

---

## ანალიზი

DLinear-მა მნიშვნელოვნად გააუმჯობესა საბაზისო (Store-Department Median) მოდელის შედეგი და Validation Set-ზე დაახლოებით **40%-იანი გაუმჯობესება** აჩვენა. ექსპერიმენტებმა აჩვენა, რომ მოდელის მუშაობაზე ყველაზე დიდი გავლენა მოახდინა Input Size-ის, Batch Size-ისა და Learning Rate-ის ცვლილებამ, ხოლო Moving Average Window-ის სწორად შერჩევამ დამატებით გააუმჯობესა პროგნოზის ხარისხი.

მიუხედავად იმისა, რომ DLinear გამოირჩევა ძალიან მარტივი არქიტექტურითა და სწრაფი ტრენინგით, საბოლოო შედეგებით იგი მაინც ჩამორჩა N-BEATS მოდელს. ეს მოსალოდნელიც იყო, რადგან DLinear მხოლოდ Trend-ისა და Seasonality-ის ხაზობრივ მოდელირებას ახდენს, მაშინ როდესაც N-BEATS უფრო ღრმა არაწრფივი არქიტექტურაა და უკეთ სწავლობს Walmart-ის გაყიდვების მონაცემებში არსებულ რთულ დამოკიდებულებებს.

თუმცა მიღებული შედეგები ადასტურებს, რომ სწორად შერჩეული ჰიპერპარამეტრების შემთხვევაში DLinear წარმოადგენს სწრაფ, სტაბილურ და კონკურენტუნარიან საბაზისო Deep Learning მოდელს დროითი რიგების პროგნოზირების ამოცანებისთვის.

| Baseline WMAE | **2244.99** |
| Improvement over Baseline | **43.1%** |

Validation WMAE არის მთავარი მოდელის შესარჩევი მეტრიკა, ხოლო train-tail WMAE გამოიყენება დამატებით backtesting შეფასებად, რათა ვნახოთ, როგორ პროგნოზირებს მოდელი ტრენინგში უნახავ უახლოეს 14 კვირას.
