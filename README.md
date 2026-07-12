# Walmart Store Sales Forecasting
Walmart-ის მაღაზიებისა და დეპარტამენტების ყოველკვირეული გაყიდვების პროგნოზირება Kaggle-ის [Walmart Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting) კონკურსის მონაცემებზე.

**MLflow ექსპერიმენტები:** [DagsHub](https://dagshub.com/izere23/ML-Final-Walmart-Recruiting-Store-Sales-Forecasting.mlflow/#/experiments)


პროექტში შედარებულია სამი ძირითადი მიდგომა:

- **Tree-based მოდელები:** LightGBM, XGBoost;
- **Deep Learning მოდელები:** N-BEATS, DLinear, Temporal Fusion Transformer (TFT);
- **კლასიკური დროითი რიგების მოდელები:** Prophet, ARIMA, Seasonal AutoARIMA (SARIMA).

ყველა მნიშვნელოვანი ექსპერიმენტი დალოგილია **MLflow/DagsHub**-ში. მოდელები შეფასებულია ერთიანი time-based validation split-ითა და კონკურსის ოფიციალური **Weighted Mean Absolute Error (WMAE)** მეტრიკით.

> **საუკეთესო local validation შედეგი:** **XGBoost — 1254.85 WMAE**  
> **საუკეთესო Deep Learning შედეგი:** **N-BEATS — 1276.74 WMAE**  
> **Kaggle-ზე შეფასებული მოდელი:** **LightGBM — 2811.66 Private WMAE**

---

## პროექტი

1. [ამოცანა და მონაცემები](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting)
2. [Exploratory Data Analysis](https://github.com/IrakliZerekidze/ML-Final-Walmart-Recruiting-Store-Sales-Forecasting/blob/main/eda-walmart.ipynb)
3. [საერთო preprocessing pipeline](https://github.com/IrakliZerekidze/ML-Final-Walmart-Recruiting-Store-Sales-Forecasting/blob/main/preprocessing.py)
4. [LightGBM](https://github.com/IrakliZerekidze/ML-Final-Walmart-Recruiting-Store-Sales-Forecasting/blob/main/model_experiment_LightGBM.ipynb)
5. [XGBoost](https://github.com/IrakliZerekidze/ML-Final-Walmart-Recruiting-Store-Sales-Forecasting/blob/main/model-experiment-xgboost.ipynb)
6. [N-BEATS](https://github.com/IrakliZerekidze/ML-Final-Walmart-Recruiting-Store-Sales-Forecasting/blob/main/model_experiment_NBEATS.ipynb)
7. [DLinear](https://github.com/IrakliZerekidze/ML-Final-Walmart-Recruiting-Store-Sales-Forecasting/blob/main/model_experiment_DLinear.ipynb)
8. [Temporal Fusion Transformer](https://github.com/IrakliZerekidze/ML-Final-Walmart-Recruiting-Store-Sales-Forecasting/blob/main/model_experiment_TFT.ipynb)
9. [Prophet](#prophet)
10. [ARIMA](https://github.com/IrakliZerekidze/ML-Final-Walmart-Recruiting-Store-Sales-Forecasting/blob/main/model_experiment_ARIMA.ipynb)
11. [Seasonal AutoARIMA](https://github.com/IrakliZerekidze/ML-Final-Walmart-Recruiting-Store-Sales-Forecasting/blob/main/model_experiment_SARIMA.ipynb)

## პროექტის სტრუქტურა

```text
ML-Final-Walmart-Recruiting-Store-Sales-Forecasting/
│
├── README.md
│   └── პროექტის მთავარი დოკუმენტაცია: ამოცანა, EDA, preprocessing,
│       მოდელების ექსპერიმენტები და საბოლოო შედეგების შედარება.
│
├── .gitignore
│   └── განსაზღვრავს ფაილებსა და საქაღალდეებს, რომლებიც GitHub-ზე
│       არ უნდა აიტვირთოს, მაგალითად მონაცემები, checkpoints და cache ფაილები.
│
├── preprocessing.py
│   └── საერთო preprocessing pipeline: მონაცემების ჩატვირთვა, გაერთიანება,
│       time-based split, weekly grid-ის შევსება და calendar feature-ების შექმნა.
│
├── eda-walmart.ipynb
│   └── მონაცემების საწყისი კვლევა: განაწილებები, missing values,
│       სეზონურობა, holiday ეფექტები, Store Type და Store Size.
│
├── model_experiment_LightGBM.ipynb
│   └── LightGBM-ის training და ექსპერიმენტები: categorical encoding,
│       target statistics, lag feature-ები, regularization და შეფასება.
│
├── model-experiment-xgboost.ipynb
│   └── XGBoost-ის ექსპერიმენტები leakage-safe feature engineering-ით,
│       L1 objective-ითა და hyperparameter tuning-ით.
│
├── model_experiment_NBEATS.ipynb
│   └── N-BEATS global univariate forecasting ექსპერიმენტები:
│       input window, learning rate, training steps და architecture tuning.
│
├── model_experiment_DLinear.ipynb
│   └── DLinear-ის ექსპერიმენტები: input size, moving-average window,
│       batch size, scaler და training duration.
│
├── model_experiment_TFT.ipynb
│   └── Temporal Fusion Transformer-ის ექსპერიმენტები calendar, static,
│       MarkDown და external covariates-ის სხვადასხვა კომბინაციით.
│
├── model_experiment_ARIMA.ipynb
│   └── ARIMA ექსპერიმენტები თითოეული Store–Department სერიისთვის,
│       სხვადასხვა order-ით, interpolation-ითა და fallback სტრატეგიით.
│
└── model_experiment_SARIMA.ipynb
    └── Seasonal ARIMA / AutoARIMA ექსპერიმენტები 52-კვირიანი
        სეზონურობის გამოყენებით.
```

## ამოცანა და მონაცემები

პროექტის მიზანია Walmart-ის სხვადასხვა მაღაზიასა და დეპარტამენტში ყოველკვირეული გაყიდვების პროგნოზირება. პროგნოზი უნდა შეიქმნას არა მხოლოდ მაღაზიის, არამედ კონკრეტული დეპარტამენტისა და კვირის დონეზე, ამიტომ ერთი observation განისაზღვრება `(Store, Dept, Date)` კომბინაციით, ხოლო საპროგნოზო ცვლადია `Weekly_Sales`.

ეს სტრუქტურა ამოცანას მნიშვნელოვნად ართულებს. მონაცემები არ წარმოადგენს ერთ დიდ დროით რიგს, რომელზეც ერთი სეზონური მოდელის მორგება იქნებოდა საკმარისი. რეალურად გვაქვს ათასობით პარალელური დროითი რიგი: მაგალითად, `Store=1, Dept=1` ერთი სერიაა, `Store=1, Dept=2` მეორე, ხოლო იგივე დეპარტამენტი სხვა მაღაზიაში უკვე ცალკე სერიას ქმნის. ამ სერიებს ერთმანეთისგან განსხვავებული მასშტაბი, ისტორიის სიგრძე, სეზონურობა და holiday-ებზე რეაქცია აქვთ.

მონაცემები ოთხი ძირითადი ფაილიდან მოდის:

| ფაილი | დანიშნულება |
|---|---|
| `train.csv` | ისტორიული გაყიდვები და target `Weekly_Sales` |
| `test.csv` | პერიოდი, რომლისთვისაც საბოლოო პროგნოზი უნდა შეიქმნას |
| `features.csv` | ეკონომიკური, კალენდარული და promotional ცვლადები |
| `stores.csv` | მაღაზიის ტიპი და ზომა |


`train.csv` და `test.csv` შეიცავს ძირითად იდენტიფიკატორებს — `Store`, `Dept`, `Date` და `IsHoliday`. Training მონაცემებში დამატებით გვაქვს `Weekly_Sales`, რომელიც მოდელებმა უნდა ისწავლონ. `features.csv` თითოეული მაღაზიისა და კვირისთვის გვაწვდის `Temperature`, `Fuel_Price`, `CPI`, `Unemployment` და `MarkDown1–MarkDown5` ცვლადებს. `stores.csv` კი აღწერს მაღაზიის ტიპსა და ზომას.

მთლიან მონაცემებში წარმოდგენილია **45 მაღაზია** და დაახლოებით **3,331 უნიკალური Store–Department დროითი რიგი**. სწორედ ამიტომ პროექტში შევადარეთ ერთმანეთისგან განსხვავებული მოდელირების სტრატეგიები. Tree-based მოდელები ყველა row-ს ერთ დიდ tabular dataset-ად ამუშავებენ; N-BEATS, DLinear და TFT ერთი global neural model-ით სწავლობენ მრავალ სერიას; Prophet, ARIMA და SARIMA კი თითოეულ სერიას დამოუკიდებლად ამუშავებენ.

## შეფასების მეტრიკა

კონკურსის ოფიციალური შეფასების მეტრიკაა **Weighted Mean Absolute Error (WMAE)**:

```text
WMAE = Σ(wᵢ × |yᵢ − ŷᵢ|) / Σwᵢ
```

სადაც `yᵢ` არის რეალური გაყიდვა, `ŷᵢ` — მოდელის პროგნოზი, ხოლო `wᵢ` observation-ის წონა. ჩვეულებრივ კვირაზე წონა არის `1`, holiday კვირაზე კი `5`.

ეს განსხვავება მნიშვნელოვანია, რადგან მოდელს ყველა კვირაზე თანაბარი პასუხისმგებლობა არ აქვს. მაგალითად, Thanksgiving-ის კვირაში დაშვებული 1,000 ერთეულის შეცდომა საბოლოო score-ზე ხუთჯერ უფრო ძლიერ მოქმედებს, ვიდრე იგივე შეცდომა ჩვეულებრივ კვირაში. შედეგად, მხოლოდ საშუალო პერიოდის კარგად პროგნოზირება საკმარისი არ არის — მოდელმა holiday spikes-იც უნდა დაიჭიროს.

Training-ის დროს ყველა მოდელში პირდაპირ WMAE loss არ გამოგვიყენებია. NeuralForecast მოდელებში loss იყო MAE, LightGBM-სა და XGBoost-ში კი საუკეთესო შედეგები L1 objective-მა მოგვცა. ეს არჩევანი მეტრიკასთან ლოგიკურად ახლოსაა, რადგან WMAE არსებითად weighted absolute error-ია. საბოლოო შეფასება ყველა შემთხვევაში ერთნაირი WMAE ფუნქციით შესრულდა, რათა სხვადასხვა ოჯახის მოდელები სამართლიანად შეგვედარებინა.

## Exploratory Data Analysis

EDA-ს მიზანი იყო მონაცემების სტრუქტურის, სეზონურობის, missing values-ისა და feature-ების შესაძლო მნიშვნელობის შეფასება მოდელირებამდე.

### 1. Missing values

ყველაზე მეტი missing value გვხვდება `MarkDown1–MarkDown5` სვეტებში. Promotional მონაცემები მხოლოდ ისტორიის ნაწილისთვის არის ხელმისაწვდომი, ამიტომ raw MarkDown values-ის პირდაპირ გამოყენებასთან ერთად შეიქმნა შესაბამისი არსებობის ინდიკატორებიც.

<!-- EDA IMAGE: notebook cell 16 -->
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/8d4acd77-a0bd-492e-b9b2-be6e967d50a5" />

### 2. Weekly Sales-ის განაწილება

`Weekly_Sales` ძლიერ **right-skewed** განაწილებას აჩვენებს: observation-ების უმეტესობა შედარებით მცირეა, თუმცა გვხვდება ძალიან მაღალი გაყიდვების პიკები. ასევე არსებობს მცირე რაოდენობის უარყოფითი მნიშვნელობები, რომლებიც სავარაუდოდ returns/refunds/adjustments-ს უკავშირდება.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/e04dfafe-a53e-4f24-9075-c6f6ff108253" />

### 3. გაყიდვები დროში

საერთო ყოველკვირეულ გაყიდვებში მკაფიოდ ჩანს წლიური სეზონურობა და ნოემბერ–დეკემბრის მაღალი პიკები. ეს ასაბუთებს წლიური lag-ების, seasonal statistical მოდელებისა და trend/seasonality-ზე ორიენტირებული neural architecture-ების გამოყენებას.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/dd4e2b49-8147-4937-8041-89c25837f06e" />


### 4. Holiday და non-holiday კვირები

Holiday კვირებში გაყიდვების განაწილება და პიკების მასშტაბი განსხვავებულია. WMAE-ის ხუთმაგი წონის გამო ამ პერიოდების სწორად პროგნოზირება კრიტიკულია.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/e48571ee-97bf-4177-83cc-0516b8710bab" />


### 5. Store Type და Store Size

Type A მაღაზიები ჯამურად ყველაზე მაღალ გაყიდვებს ქმნიან. Store Size-სა და total sales-ს შორისაც პოზიტიური კავშირია, თუმცა ეს Store-level ინფორმაციაა, ხოლო target Store–Department დონეზეა, ამიტომ ყველა მოდელში თანაბრად სასარგებლო არ აღმოჩნდა.

<table>
  <tr>
    <td>
      <img width="450"
           alt="image"
           src="https://github.com/user-attachments/assets/1c0b8ad8-9311-4661-9897-8959b0424daf" />
    </td>
    <td>
      <img width="450"
           alt="image"
           src="https://github.com/user-attachments/assets/54d1410a-d8ae-4e1b-a4c8-516d1d337f2a" />
    </td>
  </tr>
</table>


### 6. Department-ების განსხვავება

Department-ებს შორის გაყიდვების მასშტაბი მკვეთრად განსხვავდება. ამიტომ `Dept` უბრალო უწყვეტ რიცხვად არ უნდა აღვიქვათ — tree-based მოდელებში იგი categorical/identifier feature-ად დამუშავდა, ხოლო time-series მოდელებში `(Store, Dept)` წყვილმა `unique_id` შექმნა.

<!-- EDA IMAGE: notebook cell 32 -->
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/c4bfce96-3863-4422-af63-5efeff538c56" />

### 7. წლიური სეზონურობა

წლის კვირების მიხედვით საშუალო გაყიდვები არაერთგვაროვანია და წლის ბოლოს მკვეთრად იზრდება. კვირის ციკლური ბუნების გამოსახატავად შეიქმნა `Week_sin` და `Week_cos`.

<!-- EDA IMAGE: notebook cell 40 -->
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/dfc3fc67-5fa1-4114-a590-5353579c0762" />


### 8. Numeric feature-ების კორელაცია

`Temperature`, `Fuel_Price`, `CPI` და `Unemployment` target-თან მხოლოდ სუსტ პირდაპირ კორელაციას აჩვენებს. ეს არ გამორიცხავს არაწრფივ ეფექტებს, მაგრამ EDA-მ წინასწარ მიგვანიშნა, რომ calendar და historical-sales features უფრო ძლიერი signal იქნებოდა.

<!-- EDA IMAGE: notebook cell 44 -->
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/8492535d-b664-4d40-bd82-16d81b5e4bea" />

### 9. დროითი რიგების სიგრძე და უწყვეტობა

Store–Department სერიებს განსხვავებული სიგრძე აქვთ და ყველა მათგანი სრულად უწყვეტი არ არის. ეს განსაკუთრებით მნიშვნელოვანი აღმოჩნდა TFT-ისთვის, რომელსაც ყველა training series-ისთვის სრული future dataframe სჭირდება.

<!-- EDA IMAGE: notebook cell 59 -->
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/13af4b35-6fea-46f5-b5e6-ae9fc98995a6" />

### EDA-ს მთავარი დასკვნები

- გაყიდვებში ძლიერია **წლიური სეზონურობა** და holiday spikes;
- მონაცემები შედგება ათასობით განსხვავებული Store–Department სერიისგან;
- `MarkDown` feature-ები sparse და noisy-ა;
- Store-level external variables target-ის Department-level granularity-ს სრულად არ ემთხვევა;
- ისტორიული target, yearly lags და calendar features ყველაზე პერსპექტიული signal-ებია;
- validation/test პერიოდების სეზონური განსხვავება შედეგების ინტერპრეტაციისას აუცილებლად გასათვალისწინებელია.

> სურათების ჩასასმელად EDA notebook-ის შესაბამისი cell-ის output შეინახეთ მითითებული სახელით `assets/eda/` საქაღალდეში. Markdown ბმულები README-ში უკვე მომზადებულია.

---

## საერთო preprocessing pipeline

პროექტში გამოყენებული მოდელები ერთმანეთისგან მნიშვნელოვნად განსხვავდება, მაგრამ მათი შედეგების შესადარებლად აუცილებელი იყო მონაცემების ერთნაირი საწყისი დამუშავება. ამიტომ შეიქმნა საერთო `preprocessing.py` pipeline, რომელიც raw Kaggle ფაილებს ტვირთავს, აერთიანებს, ასუფთავებს და დროითი მოდელირებისთვის საჭირო სტრუქტურას ქმნის.

საერთო preprocessing-ის გამოყენებას ორი მთავარი მიზანი ჰქონდა. პირველი იყო ექსპერიმენტების გამეორებადობა: ერთსა და იმავე raw მონაცემზე სხვადასხვა notebook-ს განსხვავებული merge, date conversion ან missing-value logic არ უნდა ჰქონოდა. მეორე იყო leakage-ის კონტროლი: time-series ამოცანაში ერთი შეხედვით უწყინარმა ოპერაციამაც შეიძლება training-ში მომავლის ინფორმაცია შეიტანოს.

### Raw მონაცემების ჩატვირთვა და გაერთიანება

თავდაპირველად იტვირთება `train`, `test`, `features` და `stores` ფაილები. `Date` სვეტი ყველგან `datetime` ტიპად გარდაიქმნება, რადგან შემდგომი sorting, time split, lag-ების აგება და weekly grid სწორედ თარიღზეა დამოკიდებული.

გაყიდვების მონაცემები `features.csv`-ს უერთდება `Store`, `Date` და `IsHoliday` სვეტებით, ხოლო `stores.csv` ემატება `Store` გასაღებით. შედეგად, თითოეულ Store–Department–Week observation-ს ახლავს როგორც target-ის ისტორია, ასევე მაღაზიისა და გარე გარემოს შესახებ ინფორმაცია.

### რატომ სრულდება split grid-ის შევსებამდე

საწყის pipeline-ში სრულ weekly grid-ს ჯერ მთელ dataset-ზე ვქმნიდით და მხოლოდ შემდეგ ვყოფდით train/validation ნაწილებად. ამ მიდგომამ პირდაპირ target leakage არ შექმნა, მაგრამ training მონაცემში შეიძლებოდა ხელოვნურად გამოჩენილიყო Store–Department წყვილი, რომელიც რეალურად მხოლოდ მომავალ validation პერიოდში ჩნდებოდა. მოდელი ამ გზით მომავალი სერიის არსებობის შესახებ ინფორმაციას იღებდა.

საბოლოო leakage-safe თანმიმდევრობაა:

```text
raw data loading
→ merge
→ time-based split
→ weekly grid-ის შევსება მხოლოდ training ნაწილში
→ model-specific preprocessing
```

ამ ცვლილებამ განსაკუთრებით მნიშვნელოვანი გახადა ის ფაქტი, რომ საბოლოო შედეგები მხოლოდ corrected pipeline-ზე მიღებული run-ებიდან შევადარეთ.

### Weekly grid-ის შევსება

Walmart-ის მონაცემებში ყველა Store–Department სერია სრულ 143-კვირიან ისტორიას არ შეიცავს. ზოგიერთ სერიას შუაში ერთი ან რამდენიმე კვირა აკლია. Sequence-based მოდელებისთვის არარეგულარული დროითი ინდექსი პრობლემურია, ამიტომ `fill_grid()` თითოეული training series-ისთვის ყველა პარასკევის თარიღს ქმნის.

თუ რომელიმე კვირა raw data-ში არ არსებობდა, იქმნება ახალი row, მაგრამ `Weekly_Sales` ავტომატურად 0-ით არ ივსება. ნულით შევსება ნიშნავს მტკიცებას, რომ იმ კვირაში გაყიდვა რეალურად ნული იყო, რაც ყოველთვის სწორი არ არის. ამიტომ target თავდაპირველად `NaN` რჩება და row ინიშნება `was_grid_filled` flag-ით. შემდეგ model-specific preprocessing წყვეტს, როგორ მოექცეს ამ მნიშვნელობას. Neural და statistical მოდელებში საუკეთესო შედეგი ძირითადად linear interpolation-მა აჩვენა.

### უარყოფითი გაყიდვები

მონაცემებში გვხვდება მცირე რაოდენობის უარყოფითი `Weekly_Sales`. ისინი შეიძლება უკავშირდებოდეს დაბრუნებებს, კორექტირებებს ან refund-ებს. იმის ნაცვლად, რომ ავტომატურად წაგვეშალა ასეთი rows, შევინარჩუნეთ ორიგინალური target და დამატებით შევქმენით:

```text
is_negative_sales
Weekly_Sales_clipped = max(Weekly_Sales, 0)
```

ამან მოგვცა შესაძლებლობა, ექსპერიმენტულად შეგვედარებინა raw და clipped target. უმეტეს მოდელში ორიგინალური `Weekly_Sales` დარჩა, ხოლო საბოლოო N-BEATS არქიტექტურამ საუკეთესო შედეგი clipped target-ზე აჩვენა.

### კალენდარული feature engineering

გაყიდვებში წლიური სეზონურობისა და holiday ეფექტების გამოსახატავად შეიქმნა `Year`, `Month`, `WeekOfYear`, `Week_sin` და `Week_cos`. სინუსური და კოსინუსური encoding საჭიროა იმისთვის, რომ მოდელმა კვირის ნომერი უბრალო ხაზოვან რიცხვად არ აღიქვას: 52-ე და 1-ლი კვირა კალენდარულად ერთმანეთთან ახლოსაა, თუმცა raw რიცხვებით ერთმანეთისგან შორს ჩანს.

ასევე შეიქმნა ცალკეული holiday flag-ები:

- `is_superbowl`;
- `is_labor_day`;
- `is_thanksgiving`;
- `is_christmas`.

ეს დაყოფა უფრო ინფორმაციულია, ვიდრე მხოლოდ საერთო `IsHoliday`, რადგან სხვადასხვა დღესასწაულს გაყიდვებზე განსხვავებული ეფექტი აქვს.

### MarkDown და external feature-ები

`MarkDown1–MarkDown5` განსაკუთრებით sparse იყო. Missing value ყოველთვის არ ნიშნავს ნულოვან promotion-ს; ზოგჯერ უბრალოდ ჩანაწერი არ არსებობს. ამიტომ raw მნიშვნელობებთან ერთად დაემატა `MarkDown1_exists`–`MarkDown5_exists` ინდიკატორები.

`Temperature`, `Fuel_Price`, `CPI` და `Unemployment` preprocessing-ში შევინარჩუნეთ, თუმცა EDA-მ და შემდგომმა ექსპერიმენტებმა აჩვენა, რომ მათი signal target-თან შედარებით სუსტი იყო. განსაკუთრებით TFT-ში ყველა exogenous feature-ის ერთდროულმა დამატებამ შედეგი ვერ გააუმჯობესა.

### Model-specific preprocessing

საერთო pipeline-ის შემდეგ თითოეული მოდელი მონაცემებს საკუთარი არქიტექტურისთვის ამზადებდა. Tree-based მოდელებისთვის იქმნებოდა categorical encoding, target statistics და lag features. NeuralForecast მოდელებისთვის ფორმატი გადადიოდა `unique_id`, `ds`, `y` სტრუქტურაში. TFT-ს დამატებით სჭირდებოდა future covariates-ის სრული dataframe, ხოლო Prophet/ARIMA/SARIMA თითოეული Store–Department series-ის ცალკე დამუშავებას ახდენდნენ.

## Validation სტრატეგია

Time-series forecasting-ში random train-test split არასწორი არჩევანია, რადგან training-ში შეიძლება მოხვდეს chronologically გვიანი observation, ხოლო validation-ში — უფრო ძველი. ასეთ შემთხვევაში მოდელი პრაქტიკულად მომავლის ინფორმაციის გამოყენებით წარსულს პროგნოზირებს და მიღებული score რეალურ inference-ს აღარ ასახავს.

ამიტომ მონაცემები გავყავით თარიღის მიხედვით. Training ნაწილი მთავრდება 2012 წლის 20 ივლისს, ხოლო მომდევნო 14 კვირა validation პერიოდად რჩება:

| ნაწილი | პერიოდი |
|---|---|
| Training | 2010-02-05 → 2012-07-20 |
| Validation | 2012-07-27 → 2012-10-26 |
| Forecast horizon | 14 კვირა |

14-კვირიანი horizon ავირჩიეთ იმიტომ, რომ იგი დაახლოებით ბოლო სამ თვეს მოიცავს და ყველა მოდელისთვის ერთიანი მრავალსაფეხურიანი პროგნოზის შედარების საშუალებას იძლევა. NeuralForecast მოდელებში `h=14` პირდაპირ მოდელის პარამეტრია, ხოლო tree-based და classical მოდელებში იგივე თარიღები შეფასების ფანჯარად გამოიყენება.

Split-ის შემდეგ გამოჩნდა სერიების უწყვეტობის მნიშვნელოვანი პრობლემა. Training-ში იყო **3,321** უნიკალური Store–Department სერია, validation-ში კი **3,104**. მათგან **227** series training-ის შემდეგ აღარ ჩნდებოდა, ხოლო **10** ახალი series პირველად validation პერიოდში გამოჩნდა. გარდა ამისა, validation-ის ყველა სერიას სრული 14 observation არ ჰქონდა; მხოლოდ **2,803** series იყო სრულად წარმოდგენილი.

ეს განსხვავება მოდელებზე სხვადასხვაგვარად მოქმედებდა. N-BEATS და DLinear training-ში ცნობილ series-ებზე ავტომატურად ქმნიდნენ 14-კვირიან forecast-ს, შემდეგ კი შეფასებისას პროგნოზები რეალურად არსებულ validation rows-ს ებმებოდა. TFT-სთვის ეს საკმარისი არ იყო, რადგან future exogenous variables ყველა training series-ისა და მომდევნო 14 კვირის სრულ კომბინაციაზე სჭირდებოდა. ამიტომ TFT-სთვის ცალკე სრული future dataframe ავაგეთ.

### Validation-ის მთავარი შეზღუდვა

არჩეული validation პერიოდი 2012 წლის ივლისიდან ოქტომბრის ბოლომდე გრძელდება. მასში შეიძლება იყოს Labor Day, მაგრამ არ შედის Thanksgiving და Christmas — კონკურსის ყველაზე რთული და მაღალი წონის პერიოდები. Kaggle test horizon კი სწორედ ნოემბერ–დეკემბრის holiday-heavy კვირებს მოიცავს.

ამიტომ local validation score-ს ორი დანიშნულება აქვს:

1. ერთსა და იმავე split-ზე სხვადასხვა ექსპერიმენტის სამართლიანი შედარება;
2. preprocessing ან feature engineering ცვლილების გავლენის შეფასება.

მაგრამ local WMAE არ უნდა მივიჩნიოთ Kaggle leaderboard score-ის ზუსტ პროგნოზად. ეს განსხვავება LightGBM-ის შემთხვევაში აშკარად გამოჩნდა:

```text
Local validation WMAE: 1281.58
Kaggle Private WMAE:   2811.66
```

Local პერიოდში მოდელი კარგად მუშაობდა, მაგრამ test-ის Thanksgiving/Christmas spikes გაცილებით რთული აღმოჩნდა. საბოლოო დასკვნებისას ამიტომ ვასხვავებთ **საუკეთესო local მოდელს** და **Kaggle-ზე რეალურად შეფასებულ მოდელს**.

## Baseline

ყველა მოდელის შედეგის შესაფასებლად გვჭირდებოდა მარტივი reference, რომელსაც რთული არქიტექტურის გამოყენება არ სჭირდება. ძირითად baseline-ად ავირჩიეთ **Store–Department historical median**: თითოეული სერიის მომავალი გაყიდვა მისი training history-ის მედიანით პროგნოზირდება.

```text
Validation WMAE ≈ 2245.16
```

Median საშუალოზე უფრო გამძლეა holiday spikes-ისა და outlier-ების მიმართ. თუ კონკრეტული Store–Department series-ის ისტორია არ იყო ხელმისაწვდომი, fallback უფრო ზოგად დონეზე გადადიოდა. Baseline-ის მიზანი არ ყოფილა საუკეთესო პროგნოზის შექმნა; იგი გვიჩვენებდა, რამდენად მეტ ინფორმაციას სწავლობს მოდელი უბრალო ისტორიულ level-თან შედარებით.

ყველა საბოლოო მოდელმა, გარდა მარტივი ARIMA-ს ზოგიერთი run-ისა, baseline-ს მნიშვნელოვნად აჯობა. საუკეთესო XGBoost-მა WMAE დაახლოებით 44%-ით შეამცირა, რაც ადასტურებს, რომ yearly seasonality, hierarchical target statistics და nonlinear ურთიერთქმედებები რეალურ დამატებით signal-ს შეიცავდა.

## LightGBM

### მიდგომა

LightGBM გამოყენებულია როგორც **global tabular regressor**: თითოეული `(Store, Dept, Date)` ერთი training row-ია, ხოლო ერთი საერთო მოდელი სწავლობს ყველა series-ზე.

ძირითადი leakage-safe feature-ები:

- categorical `Store`, `Dept` და მათი კომბინაციები;
- train-only smoothed target statistics (`RichEncoder`);
- `lag_52`, `lag_104`;
- year-over-year ratio;
- calendar და holiday-distance features.

### ექსპერიმენტების განვითარება

LightGBM-ის ექსპერიმენტები ეტაპობრივად ავაგეთ. საწყის run-ში `Store` და `Dept` ჩვეულებრივ integer feature-ებად შევიტანეთ. შედეგი baseline-ზე უარესი იყო, რადგან ხე დეპარტამენტის ნომრებს რაოდენობრივ მნიშვნელობებად აღიქვამდა — თითქოს Dept 10 Dept 5-ზე „ორჯერ მეტი“ ყოფილიყო. Categorical encoding-მა ეს პრობლემა ნაწილობრივ მოაგვარა და score მკვეთრად გააუმჯობესა.

შემდეგ თითოეული Store–Department სერიის ისტორიული level target encoding-ით გადავეცით. ამ ცვლილებამ მოდელს საშუალება მისცა ერთმანეთისგან გაერჩია დიდი და პატარა დეპარტამენტები. ყველაზე დიდი შემდგომი გაუმჯობესება `lag_52`-მა მოიტანა: გასული წლის იგივე კვირის გაყიდვა Walmart-ის მონაცემებში ერთ-ერთი ყველაზე ძლიერი signal აღმოჩნდა. `lag_104`-მა მოდელს ორი წლის წინანდელი სეზონური reference-იც მისცა.

RichEncoder-ის ეტაპზე უბრალო საშუალოს ნაცვლად რამდენიმე იერარქიული train-only statistic გამოვიყენეთ. ამან sparse series-ებში ზედმეტი noise შეამცირა. მოგვიანებით დავამატეთ L1 objective და capacity/regularization tuning. საბოლოო 2,000-tree configuration ნელა სწავლობდა, მაგრამ რთული ურთიერთქმედებების დასაჭერად საკმარისი capacity ჰქონდა.

ექსპერიმენტებისას განსაკუთრებით მნიშვნელოვანი იყო RecentLags run. თავდაპირველად მან 1184.60 WMAE აჩვენა, თუმცა შემოწმებით აღმოჩნდა, რომ validation observation-ებისთვის `lag_1`, `lag_2` და `lag_4` რეალურ validation target-ს კითხულობდა. Honest recursive ვერსიაში score 1366.55-მდე გაუარესდა. ამიტომ leaked run საბოლოო შედარებიდან სრულად ამოვიღეთ.

### ძირითადი ექსპერიმენტები

| ექსპერიმენტი | Validation WMAE | შეფასება |
|---|---:|---|
| Raw numeric identifiers | 4197.61 | Store/Dept-ის რიცხვად აღქმა არასწორია |
| Categorical encoding | 2261.94 | მკვეთრი გაუმჯობესება |
| Target Encoding | 1835.45 | series-level signal დაემატა |
| + `lag_52` | 1552.86 | წლიური სეზონურობა ძლიერი feature-ია |
| + `lag_104` | 1491.87 | დამატებითი გაუმჯობესება |
| RichEncoder + YoY + Calendar | 1351.65 | ძლიერი clean baseline |
| Recent lags, leaked | 1184.60 | უარყოფილია — future actuals გამოიყენებოდა |
| Recent lags, honest recursive | 1366.55 | leakage-ის გარეშე გაუმჯობესება გაქრა |
| L1 objective | 1340.29 | WMAE-სთან უკეთესი შესაბამისობა |
| 127 leaves, unregularized | 1310.64 | კარგი score, მაგრამ დიდი overfit gap |
| + regularization | 1305.59 | gap მკვეთრად შემცირდა |
| **2000 trees, lr=0.03** | **1281.58** | **საუკეთესო სანდო LightGBM** |

### საბოლოო კონფიგურაცია

| პარამეტრი | მნიშვნელობა |
|---|---|
| Objective | `regression_l1` |
| `num_leaves` | 127 |
| `n_estimators` | 2000 |
| `learning_rate` | 0.03 |
| `min_child_samples` | 200 |
| `reg_lambda` | 10.0 |
| `subsample` / `colsample_bytree` | 0.8 / 0.8 |
| Validation WMAE | **1281.58** |

### Kaggle შედეგი

| Split | WMAE |
|---|---:|
| Public leaderboard | 2642.07 |
| Private leaderboard | **2811.66** |

Local და Kaggle შედეგებს შორის სხვაობის მთავარი მიზეზია validation/test სეზონური mismatch და holiday-heavy test horizon.

---

## XGBoost

### მიდგომა

XGBoost-შიც გამოყენებულია global tabular forecasting. საბოლოო ექსპერიმენტებში LightGBM-ის leakage-safe feature recipe გადავიტანეთ XGBoost-ზე: RichEncoder, yearly lags, YoY ratio, calendar extras, L1 objective და ძლიერი regularization.

### ექსპერიმენტების განვითარება

XGBoost-ის საწყისი ვერსიაც იმავე პრობლემას წააწყდა, რასაც LightGBM: raw integer identifiers სერიების ნამდვილ იდენტობას ვერ წარმოადგენდა. Categorical და target encoding-ის დამატებამ შედეგი გააუმჯობესა, ხოლო `lag_52` კვლავ ყველაზე მნიშვნელოვანი temporal feature აღმოჩნდა.

პირველი ხელით მორგებული XGBoost configuration 1409 WMAE-მდე მივიდა, მაგრამ LightGBM-ის საბოლოო recipe-სგან ჯერ კიდევ განსხვავდებოდა. ამიტომ ჩავატარეთ სუფთა A/B port: იგივე RichEncoder, `lag_52`, `lag_104`, YoY ratio, calendar features და L1 objective XGBoost-ზე გადავიტანეთ. ამ ცვლილებამ score 1254.85-მდე შეამცირა და პროექტის საუკეთესო local შედეგი მოგვცა.

ასევე შევინარჩუნეთ უფრო მკაცრად regularized run. მისი WMAE 1261.95 იყო — დაახლოებით 7 ერთეულით უარესი — მაგრამ generalization gap 104-დან 55-მდე შემცირდა. ეს აჩვენებს, რომ მხოლოდ საუკეთესო score-ის არჩევა ყოველთვის საკმარისი არ არის; production ან უცნობ test distribution-ზე უფრო კონსერვატიული configuration შეიძლება უსაფრთხო იყოს.

### ძირითადი ექსპერიმენტები

| ექსპერიმენტი | Validation WMAE | შეფასება |
|---|---:|---|
| Raw numeric identifiers | 3199.45 | სუსტი baseline |
| Categorical encoding | 1949.40 | მნიშვნელოვანი გაუმჯობესება |
| Target Encoding | 1748.03 | series-level ისტორია დაემატა |
| + `lag_52` | 1476.88 | ყველაზე ძლიერი საწყისი გაუმჯობესება |
| + `lag_104` | 1461.29 | მცირე დამატებითი სარგებელი |
| Hand-tuned old feature set | 1409.00 | ძველი საუკეთესო |
| **L1 + regularization** | **1254.85** | **საუკეთესო local score** |
| Tight regularization | 1261.95 | ოდნავ უარესი score, ნაკლები gap |

Short lags და current-date target aggregates უარყოფილია, რადგან ისინი ან leakage-ს ქმნიდნენ, ან future block-ზე deployable არ იყო.

### საუკეთესო კონფიგურაციები

| პარამეტრი | Best-score run | Tight run |
|---|---:|---:|
| Objective | `reg:absoluteerror` | `reg:absoluteerror` |
| `max_depth` | 8 | 8 |
| `n_estimators` | 2000 | 2000 |
| `learning_rate` | 0.03 | 0.03 |
| `min_child_weight` | 20 | 50 |
| `reg_lambda` | 10 | 20 |
| `gamma` | 0.1 | 0.5 |
| Validation WMAE | **1254.85** | **1261.95** |
| Generalization gap | 104 | 55 |

**1254.85** არის პროექტის საუკეთესო local validation შედეგი. Tight configuration ოდნავ უფრო კონსერვატიულია, თუმცა Kaggle submission ამ მოდელისთვის არ გაკეთებულა.

---

## N-BEATS

### მიდგომა

N-BEATS არის MLP ბლოკებზე დაფუძნებული deep forecasting architecture. თითოეული block ქმნის:

- `backcast`-ს — input history-ის ახსნას;
- `forecast`-ს — მომავალი პერიოდის პროგნოზს.

მოდელი გამოყენებულია **global univariate** რეჟიმში: ერთი საერთო neural network სწავლობს ყველა Store–Department series-ზე, მაგრამ input-ში იღებს მხოლოდ `unique_id`, `ds` და historical `y`-ს.

### ექსპერიმენტების განვითარება

N-BEATS-ის პირველი მიზანი იყო შეგვემოწმებინა, რამდენად ძლიერი შეიძლება იყოს pure univariate forecasting — ანუ მხოლოდ გაყიდვების ისტორიაზე დაფუძნებული მოდელი, ყოველგვარი Temperature, CPI, MarkDown ან Store Size feature-ის გარეშე. მიუხედავად feature-ების სიმცირისა, მოდელმა tree-based შედეგებთან ძალიან ახლოს მისვლა შეძლო.

საწყის architecture-ში Identity, Trend და Seasonality stacks გამოიყენებოდა. Training steps-ის გაზრდამ და learning rate-ის შემცირებამ optimization უფრო სტაბილური გახადა. Missing კვირების ნულით შევსებამ sequence-ში ხელოვნური ვარდნები შექმნა, ამიტომ linear interpolation უკეთესი აღმოჩნდა. Robust scaling-მა კი holiday spikes-ის გავლენა შეამცირა და შედეგი გააუარესა.

ყველაზე საინტერესო ცვლილება architecture sweep იყო. Identity stack-ის ამოღებამ და მოდელის Trend + Seasonality კომპონენტებზე ფოკუსირებამ შედეგი გააუმჯობესა. Polynomial degree 3-მა trend-ს მეტი მოქნილობა მისცა, harmonic basis-მა კი recurring seasonal pattern-ების სწავლა გააძლიერა. საბოლოო 1276.74 WMAE მიუთითებს, რომ Walmart-ის გაყიდვებში ისტორიული sequence თავისთავად ძალიან ინფორმაციულია.

### ექსპერიმენტები

| ცვლილება | საუკეთესო დაკვირვება |
|---|---|
| Training steps: 500 → 2000 | უფრო ხანგრძლივმა training-მა შედეგი გააუმჯობესა |
| Learning rate: 0.001 → 0.0005 | optimization უფრო სტაბილური გახდა |
| Batch size: 32 → 64 | შედეგი გაუარესდა |
| Missing target: zero → interpolation | interpolation უკეთესი აღმოჩნდა |
| Scaler: identity → robust | robust-მა holiday spikes დაასუსტა |
| Input size: 104 → 112 | დამატებითმა ისტორიამ შედეგი ვერ გააუმჯობესა |
| მეტი blocks | ზედმეტი complexity და უარესი generalization |
| Default → Trend + Seasonality stacks | მნიშვნელოვანი გაუმჯობესება |

Architecture sweep:

| არქიტექტურა | Validation WMAE |
|---|---:|
| Identity + Trend + Seasonality | 1294.83 |
| Trend + Seasonality | 1289.64 |
| Trend + Seasonality, harmonics=5 | 1276.95 |
| **Trend degree=3** | **1276.74** |
| Deeper seasonality | 1286.70 |

### საუკეთესო კონფიგურაცია

| პარამეტრი | მნიშვნელობა |
|---|---|
| Model type | Global univariate |
| Horizon | 14 |
| Input size | 104 |
| Learning rate | 0.0005 |
| Batch size | 32 |
| Max steps | 2000 |
| Loss | MAE |
| Target | `Weekly_Sales_clipped` |
| Missing target | Linear interpolation |
| Scaler | Identity |
| Stacks | Trend + Seasonality |
| Blocks | `[1, 1]` |
| Polynomial degree | 3 |
| Harmonics | 3 |
| MLP units | 1024 × 1024 |
| Validation WMAE | **1276.74** |
| Improvement over baseline | **43.13%** |

N-BEATS გახდა **საუკეთესო Deep Learning მოდელი** და overall ranking-ში მეორე ადგილი დაიკავა.

---

## DLinear

### მიდგომა

DLinear დროით რიგს moving average-ის საშუალებით ყოფს **trend** და **seasonal/residual** კომპონენტებად და ორივესთვის ცალკე linear projection-ს სწავლობს. არქიტექტურა N-BEATS-სა და TFT-ზე გაცილებით მარტივი და სწრაფია.

### ექსპერიმენტების განვითარება

DLinear-ის ექსპერიმენტებში მთავარი კითხვა იყო, რამდენად შორს შეიძლება წავიდეს ძალიან მარტივი decomposition-based architecture. მოდელი moving average-ით გამოყოფს trend-ს და residual/seasonal კომპონენტს, შემდეგ ორივეზე ხაზოვან projection-ს სწავლობს. ამიტომ მისთვის ყველაზე მნიშვნელოვანი ჰიპერპარამეტრი moving-average window აღმოჩნდა.

52-კვირიანი input window მხოლოდ ერთ წლიურ ციკლს მოიცავდა და აშკარად არასაკმარისი იყო. 104 და 112 კვირაზე მოდელმა ორი სეზონური ციკლის ნახვა შეძლო; საუკეთესო შედეგი 112 კვირაზე მივიღეთ. Moving average 13-მა trend-ისა და მოკლევადიანი variation-ის საუკეთესო ბალანსი შექმნა. ძალიან დიდი window, მაგალითად 51, რიგს ზედმეტად აგლუვებდა და მნიშვნელოვანი ცვლილებები იკარგებოდა.

DLinear-მა N-BEATS-ს ვერ აჯობა, თუმცა 1340.99 WMAE ასეთი მარტივი architecture-ისთვის ძლიერი შედეგია. იგი განსაკუთრებით საინტერესოა მაშინ, როდესაც training speed, ინტერპრეტაცია და დაბალი computational cost მნიშვნელოვანია.

### ძირითადი ექსპერიმენტები

| პარამეტრი | შემოწმებული მნიშვნელობები | საუკეთესო |
|---|---|---:|
| Input size | 52, 104, 112 | **112** |
| Moving average | 7, 13, 21, 25, 51 | **13** |
| Learning rate | 0.0005, 0.001 | **0.001** |
| Batch size | 32, 64, 128, 256 | **128** |
| Max steps | 1500, 2000, 2500, 3000 | **2500** |
| Scaler | Identity, Robust | **Robust** |
| Target | clipped, raw | **Weekly_Sales** |

### საუკეთესო კონფიგურაცია და შედეგი

| პარამეტრი | მნიშვნელობა |
|---|---|
| Input size | 112 |
| Horizon | 14 |
| Moving average window | 13 |
| Learning rate | 0.001 |
| Batch size | 128 |
| Max steps | 2500 |
| Scaler | Robust |
| Loss | MAE |
| Validation WMAE | **1340.99** |
| Holiday WMAE | 1356.24 |
| Non-holiday WMAE | 1335.12 |
| Improvement over baseline | **40.27%** |

DLinear-მა აჩვენა, რომ სწორად შერჩეული decomposition window-ით მარტივი linear architecture-იც კონკურენტუნარიანია, თუმცა რთული არაწრფივი pattern-ების სწავლაში N-BEATS-ს ჩამორჩა.

---

## Temporal Fusion Transformer

### მიდგომა

TFT აერთიანებს recurrent layers-ს, attention-ს, variable selection networks-ს, gating-სა და static/time-varying covariates-ის დამუშავებას. იგი გამოყენებულია **global multivariate** მოდელად.

Feature ჯგუფები:

- calendar/holiday: 8 feature;
- MarkDown values და existence flags;
- external: Temperature, Fuel Price, CPI, Unemployment;
- static: Size და Store Type one-hot features.

### სრული future dataframe-ის პრობლემა

TFT-ს prediction-ისთვის სჭირდება:

```text
ყველა training unique_id × მომდევნო 14 კვირა
```

Validation dataframe არასრული იყო, ამიტომ `make_future_dataframe()`-ით შეიქმნა სრული grid და დაემატა future covariates. Evaluation კვლავ მხოლოდ რეალურად არსებულ validation rows-ზე შესრულდა. Fallback-მდე missing იყო 23 prediction row, რომლებიც 10 unseen series-ს ეკუთვნოდა.

### ექსპერიმენტების განვითარება

TFT-ის გამოყენების მთავარი მოტივაცია იყო არა მხოლოდ target history-ის, არამედ წინასწარ ცნობილი future information-ის გამოყენება. თეორიულად calendar, holiday, MarkDown, store type და macroeconomic features მოდელს საშუალებას აძლევს forecast-ის მიზეზებიც გაითვალისწინოს. პრაქტიკაში კი მდიდარი feature set ყოველთვის უკეთესი არ აღმოჩნდა.

Target-only baseline შედარებით სუსტი იყო. Calendar features-ის დამატებამ შედეგი 1675.41-მდე გააუმჯობესა, რადგან კვირის სეზონურობა და holiday indicator-ები სუფთა და წინასწარ ცნობილი signal-ებია. Static Store Type/Size-ის დამატებამ score გააუარესა; სავარაუდოდ ეს ინფორმაცია Department-level განსხვავებების ასახვისთვის ზედმეტად ზოგადი იყო. MarkDown feature-ებმა sparsity და noise შეიტანეს, ხოლო ყველა exogenous variable-ის ერთდროულმა გამოყენებამ optimization უფრო რთული გახადა.

TFT-ის ექსპერიმენტები მნიშვნელოვანი უარყოფითი შედეგითაც დასრულდა: რთულ multivariate architecture-ს აუცილებლად არ აქვს უპირატესობა, თუ დამატებითი feature-ები სუსტი, sparse ან forecast horizon-ისთვის არასაიმედოა. ამ მონაცემებზე მცირე calendar-only configuration ყველაზე სტაბილური იყო.

### ექსპერიმენტები

| ექსპერიმენტი | Validation WMAE | დასკვნა |
|---|---:|---|
| Target-only, hidden 32 | 2025.54 | capacity არასაკმარისი იყო |
| Target-only, hidden 64 | 1862.73 | larger model უკეთესი |
| Initial multivariate | 1941.22 | ბევრი feature ავტომატურად არ ეხმარება |
| **Calendar only** | **1675.41** | **საუკეთესო TFT** |
| Calendar + Static | 1847.92 | coarse Store features არ დაეხმარა |
| Calendar + MarkDown | 1972.45 | sparsity/noise |
| Calendar + MarkDown tuned | 1809.40 | holiday score გაუმჯობესდა, overall არა |
| Calendar + Static + MarkDown | 2883.55 | optimization მკვეთრად გაუარესდა |
| All exogenous | 1923.88 | external signal სუსტი აღმოჩნდა |

### საუკეთესო კონფიგურაცია

| პარამეტრი | მნიშვნელობა |
|---|---|
| Input size / horizon | 104 / 14 |
| Future exogenous | 8 calendar/holiday features |
| Static features | None |
| Hidden size / heads | 64 / 4 |
| Learning rate | 0.001 |
| Batch size | 32 |
| Dropout | 0.10 |
| Max steps | 2500 |
| Scaler | Robust |
| Loss | MAE |
| Validation WMAE | **1675.41** |
| Improvement over baseline | **25.38%** |

TFT-ის მთავარი უპირატესობა — მდიდარი exogenous information — ამ dataset-ზე სრულად ვერ გამოვიყენეთ. მცირე და სუფთა calendar feature set-მა უკეთ იმუშავა, ვიდრე ყველა feature-ის ერთდროულმა დამატებამ.

---

## Prophet

### მიდგომა

Prophet გამოყენებულია **per-series** რეჟიმში: თითოეული Store–Department წყვილისთვის ცალკე მოდელი სწავლობს trend-ს, yearly seasonality-სა და holiday effects-ს.

დაახლოებით **3,062 series** წარმატებით fit-დებოდა, ხოლო მოკლე/პრობლემური series-ებისთვის გამოიყენებოდა fallback.

### ექსპერიმენტების განვითარება

Prophet-ის ექსპერიმენტები მიზნად ისახავდა trend-ის, yearly seasonality-ისა და holiday effect-ის ცალკე მოდელირებას. განსხვავებით global მოდელებისგან, აქ თითოეული Store–Department series-ისთვის დამოუკიდებელი Prophet იტრენინგებოდა. ეს მიდგომა ინტერპრეტირებადია, თუმცა მოკლე და sparse series-ებში ნაკლები ინფორმაციის გაზიარება შეუძლია.

Additive holiday model კარგი საწყისი შედეგი იყო. Holiday component-ის ამოღებამ WMAE გააუარესა, რაც მიუთითებს, რომ დღესასწაულების explicit modeling სასარგებლო იყო. Multiplicative seasonality იდეალურად უნდა გამოდგომოდა retail series-ებს, სადაც seasonal spike ხშირად series-ის level-ის პროპორციულია, მაგრამ რამდენიმე სერიაში არარეალურად დიდი forecast შეიქმნა. პროგნოზის historical maximum-ის 1.5-ჯერ მნიშვნელობაზე შეზღუდვამ extreme outputs გააკონტროლა და საუკეთესო 1428.88 WMAE მოგვცა.

### ექსპერიმენტები

| ექსპერიმენტი | Validation WMAE | შეფასება |
|---|---:|---|
| Additive + holidays | 1455.25 | ძლიერი baseline |
| Additive, no holidays | 1559.01 | holiday information სასარგებლოა |
| Multiplicative + holidays | 1536.02 | რამდენიმე exploded forecast |
| Multiplicative + damped | 1645.69 | underfit |
| Additive + damped | 1524.87 | baseline-ს ვერ აჯობა |
| Additive + loose | 1480.66 | overfit |
| **Multiplicative + capped** | **1428.88** | **საუკეთესო Prophet** |

### საბოლოო მოდელი

| პარამეტრი | მნიშვნელობა |
|---|---|
| Seasonality mode | Multiplicative |
| Holidays | Super Bowl, Labor Day, Thanksgiving, Christmas |
| Prediction cap | 1.5 × series historical maximum |
| Fallback | Series mean → global mean |
| Validation WMAE | **1428.88** |

Prediction capping-მა extreme multiplicative forecasts შეზღუდა და Prophet-ის საუკეთესო შედეგი შექმნა.

---

## ARIMA

### მიდგომა

ARIMA თითოეულ Store–Department series-ს დამოუკიდებლად ამუშავებს და მხოლოდ target-ის ისტორიას იყენებს. მოდელი აერთიანებს autoregressive, differencing და moving-average კომპონენტებს.

### ექსპერიმენტების ინტერპრეტაცია

ARIMA-ს მთავარი შეზღუდვა ის იყო, რომ მას explicit 52-კვირიანი seasonal component არ ჰქონდა. `(2,1,1)` configuration-ში differencing გამოიყენებოდა, თუმცა Walmart-ის ბევრი series აშკარა მუდმივ trend-ს არ შეიცავდა და differencing-მა სასარგებლო level information ნაწილობრივ დაკარგა. `(2,0,1)` უკეთესი აღმოჩნდა, მაგრამ 2139.43 WMAE მხოლოდ მცირედით სჯობდა median baseline-ს.

ეს შედეგი გვიჩვენებს, რომ მოკლე autoregressive dependence საკმარისი არ არის ისეთი retail მონაცემებისთვის, სადაც ერთი წლის წინანდელი იგივე კვირა განსაკუთრებით ინფორმაციულია.

### ექსპერიმენტები

| Order | Validation WMAE |
|---|---:|
| ARIMA `(2,1,1)` | 2214.93 |
| **ARIMA `(2,0,1)`** | **2139.43** |

ARIMA `(2,0,1)` baseline-ს მცირე სხვაობით აჯობა, მაგრამ წლიური სეზონურობის არქონის გამო holiday spikes და გრძელვადიანი recurring pattern-ები ვერ ისწავლა.

---

## Seasonal AutoARIMA

### მიდგომა

Seasonal AutoARIMA ARIMA-ს seasonal კომპონენტს ამატებს. Weekly მონაცემებისთვის გამოყენებულია:

```text
season_length = 52
```

AutoARIMA თითოეული series-ისთვის ავტომატურად ეძებს შესაბამის non-seasonal და seasonal order-ს.

### ექსპერიმენტების ინტერპრეტაცია

Seasonal AutoARIMA-ში 52-კვირიანი პერიოდულობის დამატებამ ჩვეულებრივ ARIMA-სთან შედარებით დაახლოებით 700 WMAE-ის გაუმჯობესება მოიტანა. საინტერესოა, რომ უბრალო Seasonal Naive baseline-იც 1467.97-მდე მივიდა: გასული წლის იგივე კვირის პირდაპირ გამეორება უკვე ძალიან ძლიერი forecast იყო. AutoARIMA-მ ამ seasonal reference-ს მოკლევადიანი autoregressive და moving-average კორექტირებები დაუმატა და შედეგი 1450.84-მდე შეამცირა.

უფრო ფართო search space-მა პრაქტიკულად იგივე score მისცა, მაგრამ computation გაზარდა. ამიტომ საბოლოოდ small stepwise search შევინარჩუნეთ — იგი თითქმის იმავე ხარისხს ნაკლები დროით იძლეოდა.

### ექსპერიმენტები

| ექსპერიმენტი | Validation WMAE |
|---|---:|
| Seasonal Naive (52) | 1467.97 |
| **Seasonal AutoARIMA — small search** | **1450.84** |
| Seasonal AutoARIMA — medium search | ≈1451 |
| Non-seasonal AutoARIMA | ≈2253 |

### საუკეთესო შედეგი

| პარამეტრი | მნიშვნელობა |
|---|---|
| Season length | 52 |
| Search | Stepwise, restricted seasonal space |
| Missing target | Linear interpolation |
| Validation WMAE | **1450.84** |
| Improvement over baseline | **35.30%** |

52-კვირიანი სეზონურობა იყო მთავარი მიზეზი, რის გამოც SARIMA-მ ჩვეულებრივ ARIMA-ს მკვეთრად აჯობა.

---

## მოდელების საბოლოო შედარება

ქვემოთ მოცემულია თითოეული მოდელის საუკეთესო **სანდო, leakage-safe local validation** შედეგი.

| ადგილი | მოდელი | ოჯახი | საუკეთესო Validation WMAE | Baseline-თან გაუმჯობესება |
|---:|---|---|---:|---:|
| **1** | **XGBoost** | Global tree-based | **1254.85** | **44.11%** |
| **2** | **N-BEATS** | Global deep learning | **1276.74** | **43.13%** |
| **3** | **LightGBM** | Global tree-based | **1281.58** | **42.92%** |
| 4 | DLinear | Global deep learning | 1340.99 | 40.27% |
| 5 | Prophet | Per-series classical | 1428.88 | 36.36% |
| 6 | Seasonal AutoARIMA | Per-series seasonal statistical | 1450.84 | 35.38% |
| 7 | TFT | Global multivariate deep learning | 1675.41 | 25.38% |
| 8 | ARIMA `(2,0,1)` | Per-series statistical | 2139.43 | 4.71% |
| — | Store–Dept median baseline | Baseline | 2245.16 | — |

### საბოლოო გამარჯვებული

## **XGBoost — 1254.85 Validation WMAE**

Local validation-ზე საუკეთესო მოდელი გახდა XGBoost. მისი წარმატება ერთი კონკრეტული ჰიპერპარამეტრით არ აიხსნება; შედეგი რამდენიმე სწორად შერჩეული კომპონენტის ერთობლიობამ შექმნა.

RichEncoder-მა თითოეულ series-სა და მის იერარქიულ ჯგუფებზე ისტორიული level მიაწოდა. `lag_52` და `lag_104` მოდელს აჩვენებდა, რა ხდებოდა იმავე სეზონურ კვირაში ერთი და ორი წლის წინ. YoY ratio აბსოლუტური გაყიდვების გარდა წლიური ზრდის ან კლების მიმართულებას ასახავდა. Calendar features მომავალშიც წინასწარ ცნობილია და inference-ზე უსაფრთხოდ გამოითვლება. L1 objective კი საბოლოო WMAE მეტრიკის absolute-error ბუნებას შეესაბამებოდა.

ასევე მნიშვნელოვანი იყო რეგულარიზაცია. მაღალი depth და 2,000 ხე მოდელს საკმარის capacity-ს აძლევდა, მაგრამ `min_child_weight`, `reg_lambda` და `gamma` ზედმეტად სპეციფიკური patterns-ის დამახსოვრებას ზღუდავდა. სწორედ ამ ბალანსმა მოგვცა **1254.85 WMAE**, რაც median baseline-ზე დაახლოებით **44.1%-ით უკეთესია**.

თუმცა საბოლოო შედეგის ინტერპრეტაციისას ორი განსხვავებული დასკვნა უნდა გავმიჯნოთ:

1. **Local validation-ის საუკეთესო მოდელია XGBoost — 1254.85 WMAE.**
2. **Kaggle-ზე რეალურად შეფასებული მოდელია LightGBM — 2811.66 Private WMAE.**

XGBoost-ს Kaggle submission score არ აქვს, ამიტომ მისი local უპირატესობა პირდაპირ leaderboard უპირატესობად არ უნდა ჩაითვალოს.

---

## მთავარი დასკვნები

### 1. ისტორიული გაყიდვები და წლიური სეზონურობა ყველაზე ძლიერი signal იყო

`lag_52`, `lag_104`, seasonal period 52 და N-BEATS-ის trend/seasonality architecture ყველაზე სტაბილურ გაუმჯობესებებს ქმნიდა.

### 2. მეტი feature ყოველთვის უკეთესი არ არის

TFT-ზე Calendar-only კონფიგურაციამ ყველა exogenous feature-ის ერთდროულ გამოყენებას მკვეთრად აჯობა. MarkDown და external economic variables sparse, noisy ან target-ის granular დონესთან შეუსაბამო აღმოჩნდა.

### 3. Leakage-სა და deployability-ს განსაკუთრებული ყურადღება სჭირდება

Recent lags, current-date aggregates და validation actuals-ზე აგებული features ხელოვნურად აუმჯობესებდა score-ს. საბოლოო შედარებაში მხოლოდ ისეთი features დარჩა, რომელთა გამოთვლაც რეალურ future inference-ზე შესაძლებელია.

### 4. Loss function-ის metric-თან შესაბამისობა მნიშვნელოვანია

LightGBM-ის `regression_l1` და XGBoost-ის `reg:absoluteerror` WMAE-ის absolute-error ბუნებას უკეთ მოერგო, განსაკუთრებით შესაბამის regularization-თან ერთად.

### 5. მარტივი მოდელი შეიძლება რთულ architecture-ს აჯობებდეს

DLinear-მა TFT-ს აჯობა, ხოლო Seasonal Naive/SARIMA TFT-სთან შედარებით კონკურენტული იყო. Model complexity მხოლოდ მაშინ არის სასარგებლო, როდესაც data signal და tuning მას ამართლებს.

### 6. ერთი validation window სრული დასკვნისთვის საკმარისი არ არის

14-კვირიანი local validation თანაბარ შედარებას უზრუნველყოფს, მაგრამ holiday-heavy Kaggle test-ის ზუსტი წარმომადგენელი არ არის. უკეთესი საბოლოო შეფასებისთვის სასურველია rolling-origin validation რამდენიმე სეზონურ ფანჯარაზე.

---

## ტექნოლოგიები

- Python, pandas, NumPy;
- scikit-learn;
- LightGBM, XGBoost;
- NeuralForecast / PyTorch;
- Prophet;
- statsforecast / AutoARIMA;
- MLflow და DagsHub;
- Google Colab;
- Kaggle.

---

**კონკურსი:** [Kaggle — Walmart Recruiting: Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting)
**DagsHub MLflow:** [MLFlow logs](https://dagshub.com/izere23/ML-Final-Walmart-Recruiting-Store-Sales-Forecasting.mlflow/#/experiments)

**შეფასების მეტრიკა:** WMAE  

