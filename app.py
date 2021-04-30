"""
画像を Streamlit 上で表示して、Pytorchのimagenetの学習済みモデルで分類
grad-cam などの可視化選べる
https://towardsdatascience.com/create-an-image-classification-web-app-using-pytorch-and-streamlit-f043ddf00c24

Usage:
    $ conda activate lightning
    $ streamlit run ./app.py
"""

# ==============================================
# https://arakan-pgm-ai.hatenablog.com/entry/2019/12/21/110000
# imagenetのクラス名を日本語に変換
# ==============================================
import re


class Ilsvrc2012Japanese:
    def __init__(self):
        self.jdic = {
            "bassoon": "ファゴット",
            "oboe": "オーボエ",
            "flute": "フルート",
            "bow": "弓",
            "ski": "スキー",
            "sax": "サックス",
            "crane": "クレーン",
            "hairslide": "髪留め",
            "letteropener": "ペーパーナイフ",
            "chainsaw": "チェーンソウ",
            "quill": "羽ペン",
            "hook": "フック",
            "stretcher": "ストレッチャー",
            "corkscrew": "コルク抜き",
            "fountainpen": "万年筆",
            "rifle": "ライフル",
            "electricguitar": "エレキギター",
            "paintbrush": "刷毛",
            "whistle": "ホイッスル",
            "pole": "ポール",
            "ballpoint": "ボールペン",
            "syringe": "注射器",
            "panpipe": "パンパイプ",
            "hatchet": "手斧",
            "chain": "鎖",
            "scabbard": "鞘",
            "cleaver": "包丁",
            "rubbereraser": "消しゴム",
            "missile": "ミサイル",
            "balancebeam": "平均台",
            "screwdriver": "ドライバー",
            "necklace": "ネックレス",
            "reel": "リール",
            "racket": "ラケット",
            "rule": "ルール",
            "projectile": "発射体",
            "banjo": "バンジョー",
            "swing": "スイング",
            "flagpole": "旗竿",
            "lipstick": "口紅",
            "sunglass": "サングラス",
            "drumstick": "ドラムスティック",
            "assaultrifle": "アサルトライフル",
            "binder": "バインダー",
            "broom": "ほうき",
            "envelope": "封筒／エンベロープ",
            "plunger": "ラバーカップ（トイレで使うスッポン）",
            "totempole": "トーテムポール",
            "warplane": "軍用機",
            "nail": "爪",
            "cassette": "カセット",
            "mountaintent": "登山用テント",
            "drum": "ドラム",
            "pick": "ピック",
            "acousticguitar": "アコースティックギター",
            "harmonica": "ハーモニカ",
            "microphone": "マイクロフォン",
            "sunglasses": "サングラス",
            "umbrella": "傘",
            "lighter": "ライター",
            "screw": "スクリュー",
            "plow": "すき",
            "notebook": "ノート",
            "redwine": "赤ワイン",
            "trombone": "トロンボーン",
            "wreck": "残骸・難破船など",
            "parachute": "パラシュート",
            "powerdrill": "パワードリル",
            "radio": "無線",
            "hammer": "ハンマー",
            "mortarboard": "モルタルボード",
            "crutch": "松葉杖",
            "horizontalbar": "鉄棒",
            "foldingchair": "折り畳み椅子",
            "harddisc": "ハードディスク",
            "trimaran": "３つの船体をつないだボート（三脚艇）",
            "shovel": "シャベル",
            "schooner": "スクーナー",
            "muzzle": "銃口",
            "wing": "翼",
            "gondola": "ゴンドラ",
            "harp": "ハープ",
            "violin": "バイオリン",
            "ladle": "おたま（調理道具）",
            "stupa": "仏舎利塔",
            "buckle": "バックル",
            "canopener": "缶切り",
            "loupe": "ルーペ",
            "walkingstick": "歩行用杖",
            "drillingplatform": "ボーリング基地",
            "accordion": "アコーディオン",
            "analogclock": "アナログ時計",
            "spatula": "へら",
            "swab": "綿棒",
            "stage": "ステージ",
            "cliff": "崖",
            "carpenterskit": "大工道具",
            "pirate": "海賊",
            "wallet": "財布",
            "gar": "ガー（魚）",
            "bolotie": "ボータイ（ネクタイ）",
            "submarine": "潜水艦",
            "bowtie": "蝶ネクタイ",
            "pencilbox": "筆箱",
            "centipede": "ムカデ",
            "vacuum": "真空",
            "dragonfly": "トンボ",
            "puck": "パック",
            "airship": "飛行船",
            "torch": "トーチ",
            "digitalwatch": "デジタル腕時計",
            "abaya": "アバヤ（頭からつま先まで覆うゆったりしたローブ）",
            "tobaccoshop": "煙草屋",
            "knot": "結び目",
            "remotecontrol": "リモコン",
            "website": "ウエブサイト",
            "matchstick": "マッチ棒",
            "ocarina": "オカリナ",
            "bearskin": "熊皮",
            "pencilsharpener": "鉛筆削り",
            "guillotine": "ギロチン",
            "digitalclock": "デジタル時計",
            "bobsled": "ボブスレー",
            "dogsled": "犬ぞり",
            "paddle": "パドル",
            "harvester": "収穫機",
            "aircraftcarrier": "空母",
            "snorkel": "スノーケル",
            "beeeater": "蜂食鳥",
            "sleepingbag": "寝袋",
            "cornet": "コルネット",
            "cellulartelephone": "携帯電話",
            "safetypin": "安全ピン",
            "sandal": "サンダル",
            "facepowder": "フェースパウダー",
            "parallelbars": "平行棒",
            "dock": "ドック（船着き場）",
            "cinema": "映画",
            "whitestork": "鳥（シュバシコウ／コウノトリ）",
            "theatercurtain": "緞帳（映画館のカーテン）",
            "tileroof": "瓦屋根",
            "doormat": "ドアマット",
            "chainmail": "鎖帷子",
            "canoe": "カヌー",
            "containership": "コンテナ船",
            "blackstork": "鳥（ナベコウ／コウノトリ）",
            "jean": "ジーンズ",
            "militaryuniform": "軍服",
            "maillot": "マイヨ",
            "maraca": "マラカス",
            "prayerrug": "礼拝用マット",
            "fireboat": "消防艇",
            "kite": "凧",
            "tennisball": "テニスボール",
            "punchingbag": "パンチングバッグ",
            "spinylobster": "イセエビ",
            "purse": "財布",
            "marimba": "マリンバ",
            "packet": "パケット",
            "modem": "モデム",
            "mailbag": "郵袋",
            "gong": "ゴング",
            "wig": "かつら",
            "thresher": "脱穀機",
            "tractor": "トラクター",
            "mousetrap": "ネズミ捕り",
            "ibex": "アイベックス（ヤギ）",
            "trafficlight": "信号機",
            "lawnmower": "芝刈り機",
            "woodenspoon": "木製スプーン",
            "hamper": "妨げる",
            "spaceshuttle": "スペースシャトル",
            "chime": "チャイム",
            "hay": "干し草",
            "spindle": "スピンドル",
            "damselfly": "わがままに",
            "tank": "タンク",
            "laptop": "ノートパソコン",
            "switch": "スイッチ",
            "winebottle": "ワインボトル",
            "trailertruck": "トレーラートラック",
            "revolver": "リボルバー",
            "crayfish": "ザリガニ",
            "featherboa": "羽根の襟巻",
            "wallclock": "壁時計",
            "bulbul": "電球",
            "toucan": "オオハシ",
            "airliner": "旅客機",
            "lakeside": "湖畔",
            "scorpion": "サソリ",
            "academicgown": "アカデミックガウン",
            "cello": "チェロ",
            "strainer": "ストレーナー",
            "birdhouse": "巣箱",
            "cowboyhat": "カウボーイハット",
            "frenchhorn": "フレンチホルン",
            "nematode": "線虫",
            "towtruck": "レッカー車",
            "breakwater": "防波堤",
            "rockingchair": "ロッキングチェア",
            "schipperke": "スキッパーキ（犬）",
            "loudspeaker": "拡声器",
            "streetsign": "交通標識",
            "plane": "飛行機",
            "fireengine": "消防車",
            "go-kart": "ゴーカート",
            "padlock": "南京錠",
            "cassetteplayer": "カセット・プレーヤー",
            "cannon": "大砲",
            "snowmobile": "スノーモービル",
            "stethoscope": "聴診器",
            "clumber": "クランバー",
            "barbell": "バーベル",
            "sussexspaniel": "サセックス・スパニエル（犬）",
            "carton": "カートン",
            "coucal": "クーカル",
            "pretzel": "プレッツェル",
            "lifeboat": "救命ボート",
            "maypole": "メイポール（スェーデンの祭りで使うポール）",
            "catamaran": "船（双胴船／カタマラン）",
            "bikini": "ビキニ",
            "balloon": "バルーン",
            "hairspray": "ヘアスプレー",
            "buckeye": "バックアイ",
            "croquetball": "クロケットのボール",
            "housefinch": "メキシコマシコ（鳥）",
            "tripod": "三脚",
            "cloak": "マント",
            "beerbottle": "ビール瓶",
            "hand-heldcomputer": "携帯情報端末",
            "bandaid": "バンドエイド",
            "scoreboard": "スコアボード",
            "weasel": "イタチ",
            "sealyhamterrier": "シーリハムテリア（犬）",
            "tick": "ダニ",
            "barrow": "手押し車",
            "jinrikisha": "人力車",
            "alp": "アルプ",
            "commonnewt": "スベイモリ（イモリ）",
            "norwichterrier": "ノーリッチテリア（犬）",
            "cradle": "ゆりかご",
            "ambulance": "救急車",
            "brittanyspaniel": "ブリタニースパニエル（犬）",
            "solardish": "ソーラーディッシュ（パラボラ）",
            "alligatorlizard": "ゲルホノタス（アシナシトカゲ）",
            "suspensionbridge": "吊り橋",
            "magneticcompass": "方位磁針",
            "studiocouch": "3人掛けソファ",
            "spottedsalamander": "ギボシサンショウウオ",
            "kingsnake": "キングスネーク",
            "scotchterrier": "スコティッシュ・テリア（犬）",
            "velvet": "ベルベット",
            "brassiere": "ブラジャー",
            "stole": "ストール",
            "combinationlock": "組み合わせロック",
            "iron": "鉄",
            "wool": "ウール",
            "sombrero": "ソンブレロ",
            "fryingpan": "フライパン",
            "long-hornedbeetle": "カミキリムシ",
            "seashore": "海岸",
            "boathouse": "ボートハウス",
            "mountainbike": "マウンテンバイク",
            "yawl": "ヨール（小型帆船）",
            "baldeagle": "白頭ワシ",
            "thimble": "指ぬき",
            "thatch": "かやぶき",
            "americanlobster": "アメリカ人",
            "bandedgecko": "コレオニー（トカゲモドキ）",
            "lumbermill": "製材所",
            "bookjacket": "ブックカバー",
            "kingcrab": "タラバガニ",
            "albatross": "アルバトロス",
            "oystercatcher": "オイスターキャッチャー",
            "bathingcap": "スイムキャップ",
            "beacon": "ビーコン",
            "tapeplayer": "カセットプレイヤー",
            "cockroach": "ゴキブリ",
            "chocolatesauce": "チョコレートソース",
            "abacus": "そろばん",
            "ibizanhound": "イビザンハウンド（犬）",
            "spoonbill": "スプーンビル",
            "handblower": "ハンドブロワー",
            "pooltable": "ビリヤード台",
            "bellcote": "鐘楼",
            "germanshepherd": "ジャーマンシェパード（犬）",
            "labradorretriever": "ラブラドール・レトリバー（犬）",
            "cuirass": "キュイラス（紅土）",
            "bottlecap": "瓶のキャップ",
            "halftrack": "半トラック",
            "mouse": "マウス",
            "clog": "下駄",
            "goldenretriever": "ゴールデンレトリバー（犬）",
            "shield": "シールド",
            "borzoi": "ボルゾイ（犬）",
            "unicycle": "一輪車",
            "grasshopper": "バッタ",
            "tablelamp": "電気スタンド",
            "perfume": "香水",
            "eft": "ｅｆｔ",
            "ant": "蟻",
            "speedboat": "スピードボート",
            "tricycle": "三輪車",
            "racer": "レーサー",
            "pier": "桟橋",
            "jigsawpuzzle": "ジグソーパズル",
            "welshspringerspaniel": "ウェルシュ・スプリンガー・スパニエル（犬）",
            "mask": "マスク",
            "chest": "胸",
            "barn": "納屋",
            "jersey": "ジャージー",
            "vulture": "ハゲタカ",
            "australianterrier": "オーストラリアン・テリア（犬）",
            "throne": "王位／王座／王様が座るような豪華な椅子",
            "baseball": "野球",
            "monastery": "修道院",
            "swimmingtrunks": "海パン",
            "cowboyboot": "カウボーイブーツ",
            "quilt": "キルト",
            "dumbbell": "ダンベル",
            "holster": "ホルスター",
            "liner": "ライナー",
            "snowplow": "除雪車",
            "grandpiano": "グランドピアノ",
            "forklift": "フォークリフト",
            "appenzeller": "アッペンツェル",
            "steeldrum": "スチールドラム",
            "breastplate": "胸当て",
            "gown": "ガウン",
            "papillon": "パピヨン",
            "shoeshop": "靴屋",
            "windsortie": "ウィンザー・タイ（ネクタイ）",
            "gordonsetter": "ゴードンセッター",
            "agama": "トカゲ（アガマ）",
            "toaster": "トースター",
            "barnspider": "オニグモ",
            "eskimodog": "エスキモードッグ（犬）",
            "menu": "メニュー",
            "runningshoe": "ランニングシューズ",
            "hognosesnake": "ヘビ（ホングノーズスネーク）",
            "promontory": "岬",
            "irishwolfhound": "アイリッシュウルフハウンド（犬）",
            "cdplayer": "cdプレーヤー",
            "kuvasz": "クーバース（犬）",
            "tray": "トレイ",
            "firescreen": "ファイアースクリーン",
            "scale": "定規",
            "viaduct": "高架橋",
            "bernesemountaindog": "バーニーズ・マウンテン・ドッグ（犬）",
            "shoppingcart": "ショッピングカート",
            "admiral": "提督",
            "dalmatian": "ダルメシアン",
            "passengercar": "乗用車",
            "binoculars": "双眼鏡",
            "spacebar": "スペースキー",
            "four-poster": "四柱",
            "irishsetter": "犬（アイリッシュセッター）",
            "waterbottle": "ウォーターボトル",
            "ptarmigan": "雷鳥",
            "pickelhaube": "衛兵の兜",
            "sock": "靴下",
            "skimask": "目出し帽",
            "sweatshirt": "トレーナー",
            "golfball": "ゴルフボール",
            "cardigan": "カーディガン",
            "hip": "お尻",
            "file": "ファイル",
            "rugbyball": "ラグビーボール",
            "steelarchbridge": "鉄製アーチ橋",
            "corn": "コーン",
            "wire-hairedfoxterrier": "ワイアー・フォックス・テリア（犬）",
            "cab": "タクシー",
            "wok": "中華鍋",
            "typewriterkeyboard": "タイプライターキーボード",
            "bakery": "ベーカリー",
            "printer": "プリンター",
            "bannister": "階段などの手すり／バニスター",
            "wardrobe": "ワードローブ",
            "macaw": "鳥（コンゴウインコ）",
            "waffleiron": "ワッフル焼き型",
            "goldfinch": "鳥（ゴールドフィンチ）",
            "recreationalvehicle": "rv車",
            "barracouta": "魚（バラコータ／バラクーダ）",
            "handkerchief": "ハンカチ",
            "valley": "谷",
            "bordercollie": "犬（ボーダーコリー）",
            "englishsetter": "イングリッシュ・セター（犬）",
            "greaterswissmountaindog": "大きいスイスの水玉",
            "norfolkterrier": "ノーフォーク・テリア（犬）",
            "hotdog": "ホットドッグ",
            "restaurant": "レストラン",
            "nightsnake": "ヘビ（ナイトスネーク）",
            "oxcart": "牛車",
            "otterhound": "カワウソ",
            "groom": "新郎",
            "monitor": "モニター",
            "kneepad": "膝パッド",
            "germanshort-hairedpointer": "ジャーマン・ショートヘアー・シェパード（犬）",
            "showercurtain": "シャワーカーテン",
            "rockcrab": "ヨーロッパ・イチョウ・カニ",
            "electricray": "電車",
            "popbottle": "ポップなボトル",
            "junco": "鳥（ユキヒメドリ）",
            "coil": "コイル",
            "honeycomb": "ハニカム",
            "groundbeetle": "オサムシ（昆虫）",
            "neckbrace": "ネックブレス（ギプス）",
            "refrigerator": "冷蔵庫",
            "mailbox": "メールボックス",
            "palace": "宮殿",
            "englishspringer": "イングリッシュ・スプリンガー・スパニエル（犬）",
            "groenendael": "ベルジアン・シェパード・ドッグ・グローネンダール(犬）",
            "rhinocerosbeetle": "カブトムシ",
            "mink": "ミンク",
            "irishterrier": "アイリッシュ・テリア（犬）",
            "hometheater": "ホームシアター",
            "mosque": "モスク",
            "joystick": "ジョイスティック",
            "christmasstocking": "クリスマスの靴下",
            "suit": "スーツ",
            "pillow": "枕",
            "toyshop": "玩具屋",
            "black-footedferret": "クロアシイタチ",
            "dishwasher": "食器洗い機",
            "pot": "ポット",
            "ballplayer": "野球選手",
            "chickadee": "鳥（チカディー）",
            "airedale": "犬（エアデール）",
            "oscilloscope": "オシロスコープ",
            "cricket": "クリケット",
            "hourglass": "砂時計",
            "brambling": "アトリ（鳥）",
            "crosswordpuzzle": "クロスワードパズル",
            "beaker": "ビーカー",
            "volcano": "火山",
            "europeanfiresalamander": "ファイアサラマンダー",
            "magpie": "カササギ（鳥）",
            "doberman": "ドーベルマン（犬）",
            "castle": "城",
            "blackandgoldgardenspider": "キマダラコガネグモ",
            "movingvan": "引っ越しトラック",
            "apron": "エプロン",
            "vizsla": "ショートヘアード・ハンガリアン・ビズラ（犬）",
            "yorkshireterrier": "ヨークシャーテリア（犬）",
            "convertible": "自動車（コンバーチブル／オープンカー）",
            "boaconstrictor": "ボアコンストリクター（ヘビ）",
            "hummingbird": "ハチドリ（鳥）",
            "lakelandterrier": "レークランド・テリア（犬）",
            "computerkeyboard": "キーボード",
            "carousel": "メリーゴーラウンド",
            "sunscreen": "日焼け止め",
            "malinois": "ベルジアン・シェパード・ドッグ・マリノア（犬）",
            "westhighlandwhiteterrier": "ウエスト・ハイランド・ホワイト・テリア（犬）",
            "mobilehome": "移動住宅",
            "mantis": "カマキリ",
            "banana": "バナナ",
            "loafer": "ローファー",
            "rotisserie": "ロティサリー（肉をあぶる器具）",
            "diskbrake": "ディスクブレーキ",
            "shoppingbasket": "買い物カゴ",
            "basset": "バセット・ハウンド（犬）",
            "maze": "迷路",
            "rottweiler": "ロットワイラー（犬）",
            "ipod": "ipod",
            "newfoundland": "ニューファンドランド島",
            "mitten": "ミトン",
            "siberianhusky": "シベリアンハスキー（犬）",
            "steamlocomotive": "蒸気機関車",
            "collie": "コリー（犬）",
            "ladybug": "てんとう虫",
            "sliderule": "計算尺",
            "coho": "ギンザケ",
            "ear": "耳",
            "library": "図書館",
            "comicbook": "コミックブック",
            "television": "テレビ",
            "barrel": "バレル",
            "icelolly": "アイスキャンディー",
            "icecream": "アイスクリーム",
            "sturgeon": "チョウザメ",
            "africangrey": "ヨウム（鳥）",
            "grocerystore": "食料品店",
            "sarong": "サロン",
            "barometer": "バロメーター",
            "miniskirt": "ミニスカート",
            "curly-coatedretriever": "カーリーコーテッド・レトリーバー（犬）",
            "limousine": "リムジン",
            "hen-of-the-woods": "キノコ（マイタケ）",
            "soccerball": "サッカーボール",
            "sandbar": "砂州",
            "macaque": "マカク猿",
            "kelpie": "ケルピー（牧羊犬）",
            "vestment": "ベスト（祭服）",
            "freightcar": "貨車",
            "redshank": "アカアシシギ（鳥）",
            "thundersnake": "ヘビ（雷蛇）",
            "streetcar": "路面電車",
            "bonnet": "ボンネット",
            "caldron": "大釜",
            "chainlinkfence": "ダイヤ網の金網",
            "seasnake": "ウミヘビ",
            "arabiancamel": "ヒトコブラクダ",
            "pinwheel": "風車",
            "desk": "机",
            "confectionery": "お菓子",
            "lorikeet": "ヒインコ（鳥）",
            "bathtub": "バスタブ",
            "paddlewheel": "パドルホイール（外輪）",
            "chihuahua": "チワワ（犬）",
            "lampshade": "ランプシェード",
            "seaurchin": "うに",
            "blackwidow": "クロゴケグモ",
            "jay": "カケス（鳥）",
            "flatworm": "フラットワーム（扁形動物）",
            "cocktailshaker": "シェイカー",
            "crib": "ベビーベッド",
            "weimaraner": "ワイマラナー（犬）",
            "bostonbull": "ボストンテリア（犬）",
            "bulletproofvest": "防弾チョッキ",
            "basketball": "バスケットボール",
            "whiptail": "ホイップテイル（トカゲ）",
            "platerack": "プレート用ラック",
            "meatloaf": "ミートローフ",
            "bucket": "バケツ",
            "screen": "画面",
            "labcoat": "白衣",
            "kimono": "着物",
            "giantschnauzer": "ジャイアントシュナウザー（犬）",
            "stonewall": "石垣",
            "orangutan": "オランウータン",
            "motorscooter": "スクーター",
            "safe": "安全",
            "furcoat": "毛皮のコート",
            "pembroke": "ペンブローク城",
            "stopwatch": "ストップウォッチ",
            "hoopskirt": "フープスカート",
            "chesapeakebayretriever": "チェサピークベイリトリバー（犬）",
            "isopod": "等脚（ワラジムシ）",
            "crate": "クレート（木箱）",
            "hamster": "ハムスター",
            "ruddyturnstone": "キョウジョシギ（鳥）",
            "entlebucher": "エントレブッハー・キャトル・ドッグ（犬）",
            "flat-coatedretriever": "フラットコートリトリーバー（犬）",
            "cockerspaniel": "イングリッシュ・コッカー・スパニエル（犬）",
            "polecat": "ヨーロッパケナガイタチ",
            "photocopier": "コピー機",
            "italiangreyhound": "グレイハウンド（犬）",
            "arcticfox": "ホッキョクギツネ",
            "candle": "キャンドル",
            "rockpython": "ヘビ（ニシキヘビ）",
            "spaceheater": "スペースヒーター（暖房器具）",
            "shoji": "障子",
            "gardenspider": "キマダラコガネグモ",
            "scottishdeerhound": "スコティッシュディアハウンド（犬）",
            "malamute": "アラスカン・マラミュート（犬）",
            "piggybank": "貯金箱",
            "ashcan": "ゴミ捨て缶",
            "gaspump": "ガスポンプ",
            "sewingmachine": "ミシン",
            "upright": "アップライト",
            "blenheimspaniel": "ブレナム・スパニエル（犬）",
            "irishwaterspaniel": "アイルランドの水辺",
            "coralfungus": "サンゴのような菌類（シロソウメンタケ科）",
            "washer": "座金",
            "conch": "巻き貝",
            "polaroidcamera": "ポラロイドカメラ",
            "americanegret": "鳥（ダイサギ）",
            "japanesespaniel": "狆（チン＝犬）",
            "poncho": "ポンチョ",
            "vendingmachine": "自動販売機",
            "gilamonster": "アメリカドクトカゲ",
            "guineapig": "モルモット",
            "walkerhound": "ツリーイング・ウォーカー・クーンハウンド（犬）",
            "beaver": "ビーバー",
            "pajama": "パジャマ",
            "church": "教会",
            "backpack": "バックパック",
            "capuchin": "猿（オマキザル）",
            "africanchameleon": "アフリカンカメレオン",
            "measuringcup": "計量カップ",
            "ruffedgrouse": "エリマキライチョウ（鳥）",
            "bookshop": "書店",
            "slug": "ナメクジ",
            "dingo": "ディンゴ（タイリクオオカミ）",
            "cucumber": "きゅうり",
            "pizza": "ピザ",
            "dam": "ダム",
            "englishfoxhound": "イングリッシュ・フォックスハウンド（犬）",
            "frilledlizard": "エリマキトカゲ",
            "llama": "ラマ",
            "indigobunting": "ルリノジコ（鳥）",
            "soft-coatedwheatenterrier": "ソフトコーテッド・ウィートン・テリア（犬）",
            "ox": "牛",
            "gazelle": "ガゼル",
            "coffeemug": "コーヒーマグカップ",
            "bib": "よだれかけ",
            "samoyed": "サモエド（犬）",
            "vase": "花瓶",
            "grannysmith": "リンゴ（グラニースミス）",
            "staffordshirebullterrier": "スタッフォードシャー・ブル・テリア（犬）",
            "bubble": "バブル",
            "minibus": "ミニバス",
            "greatdane": "グレートデーン（犬）",
            "ping-pongball": "ピンポンボール",
            "seatbelt": "シートベルト",
            "saintbernard": "セントバーナード（犬）",
            "garbagetruck": "ごみ収集車",
            "cheetah": "チーター",
            "diningtable": "ダイニングテーブル",
            "slot": "スロット",
            "parkbench": "公園のベンチ",
            "pillbottle": "錠剤瓶",
            "strawberry": "イチゴ",
            "borderterrier": "ボーダーテリア（犬）",
            "minivan": "ミニバン",
            "indiancobra": "ヘビ（インディアンコブラ）",
            "pickup": "ピックアップトラック",
            "schoolbus": "スクールバス",
            "volleyball": "バレーボール",
            "showercap": "シャワーキャップ",
            "butchershop": "精肉店",
            "wormfence": "ワームフェンス（柵）",
            "tarantula": "タランチュラ",
            "dialtelephone": "ダイアル電話",
            "cup": "カップ",
            "headcabbage": "キャベツ",
            "broccoli": "ブロッコリ",
            "seacucumber": "ナマコ",
            "hare": "野ウサギ",
            "bloodhound": "犬（ブラッドハウンド）",
            "crashhelmet": "バイク用衝撃吸収ヘルメット",
            "americanalligator": "ワニ（アメリカンアリゲーター）",
            "pelican": "ペリカン",
            "beachwagon": "ビーチワゴン",
            "whitewolf": "白いオオカミ",
            "toiletseat": "便座",
            "pomegranate": "ザクロ",
            "dome": "ドーム",
            "wolfspider": "オオカミ",
            "papertowel": "ペーパータオル",
            "burrito": "ブリトー（食べ物）",
            "saluki": "サルキ（犬）",
            "diamondback": "ダイヤモンドバック（バイク）",
            "whippet": "ウィペット（犬）",
            "maltesedog": "マルチーズ（犬）",
            "plate": "プレート",
            "eel": "うなぎ",
            "black-and-tancoonhound": "ブラック・アンド・タン・クーンハウンド（犬）",
            "bicycle-built-for-two": "二人用自転車",
            "trolleybus": "トロリーバス",
            "triceratops": "トリケラトプス",
            "pomeranian": "ポメラニアン（犬）",
            "meerkat": "ミーアキャット",
            "oilfilter": "オイルフィルター",
            "trifle": "トライフル（デザート）",
            "monarch": "蝶（オオカバマダラ）",
            "dandiedinmont": "ダンディ・ディンモント・テリア()犬）",
            "greatpyrenees": "ピレネー山脈",
            "pedestal": "胸像などの台",
            "gasmask": "ガスマスク",
            "tub": "浴槽",
            "overskirt": "オーバースカート",
            "rapeseed": "菜種",
            "lenscap": "レンズキャップ",
            "oldenglishsheepdog": "オールド・イングリッシュ・シープドッグ（犬）",
            "pug": "パグ（犬）",
            "picketfence": "杭柵",
            "quail": "ウズラ（鳥）",
            "radiator": "ラジエーター",
            "toyterrier": "イングリッシュ･トイ･テリア（犬）",
            "coralreef": "サンゴ礁",
            "bedlingtonterrier": "ベドリントン・テリア（犬）",
            "bullettrain": "新幹線",
            "projector": "プロジェクター",
            "hen": "鳥（めんどり）",
            "keeshond": "キースホンド（犬）",
            "moped": "原付",
            "spiderweb": "クモの網",
            "carwheel": "自動車のホイール",
            "miniatureschnauzer": "ミニチュア・シュナウザー（犬）",
            "beagle": "ビーグル（犬）",
            "impala": "インパラ",
            "microwave": "電子レンジ",
            "shetlandsheepdog": "シェットランドシープドッグ（犬）",
            "lacewing": "クサカゲロウ",
            "spotlight": "スポットライト",
            "brass": "真鍮",
            "afghanhound": "アフガンハウンド（犬）",
            "lotion": "ローション",
            "barbershop": "理髪店",
            "leafhopper": "ヨコバイ（バッタ）",
            "prairiechicken": "ソウゲンライチョウ",
            "barberchair": "床屋の椅子",
            "ringnecksnake": "リングネックスネーク（ヘビ）",
            "redfox": "アカギツネ",
            "waterjug": "水差し",
            "timberwolf": "タイリクオオカミ",
            "prison": "刑務所",
            "blackswan": "ブラックスワン（鳥）",
            "redbone": "レッドボーン",
            "cliffdwelling": "マニトウ遺跡",
            "desktopcomputer": "デスクトップコンピューター",
            "odometer": "スピードメーター",
            "electriclocomotive": "電気自動車",
            "bittern": "サンカノゴイ(ペリカン科の鳥）",
            "starfish": "ヒトデ",
            "gartersnake": "ガーターヘビ",
            "apiary": "養蜂場",
            "dishrag": "布巾（お皿拭き）",
            "horsecart": "馬車",
            "egyptiancat": "エジプト猫",
            "sportscar": "スポーツカー",
            "americanchameleon": "アメリカンカメレオン",
            "oxygenmask": "酸素マスク",
            "littleblueheron": "サギ科の鳥（ヒメアカクロサギ）",
            "rhodesianridgeback": "ローデシアン・リッジバック（犬）",
            "medicinechest": "薬箱",
            "watersnake": "蛇",
            "porcupine": "ヤマアラシ",
            "siamesecat": "シャム猫",
            "greywhale": "クジラ（コククジラ）",
            "stinkhorn": "キノコ（スッポンタケ目）",
            "stove": "レンジ",
            "hammerhead": "ハンマーヘッド",
            "armadillo": "アルマジロ",
            "carmirror": "カーミラー",
            "hog": "豚",
            "eggnog": "エッグノッグ",
            "badger": "狸",
            "bassinet": "バシネット",
            "blackgrouse": "ライチョウ科の鳥（クロライチョウ）",
            "greatgreyowl": "フクロウ（カラフトフクロウ）",
            "hornedviper": "ヘビ（サハラツノクサリヘビ）",
            "basenji": "犬（バセンジー）",
            "brabancongriffon": "犬（プチ・ブラバンソン）",
            "brownbear": "ヒグマ",
            "commoniguana": "イグアナ",
            "komondor": "犬（コモンドル）",
            "sidewinder": "ヘビ（ガラガラヘビ）",
            "dutchoven": "ダッチオーブン",
            "jacamar": "キツツキ目の鳥（キリハシ）",
            "yurt": "モンゴル風の小屋（ユルト）",
            "watertower": "給水塔",
            "foxsquirrel": "リス（キツネリス）",
            "marmot": "マーモット（ネズミ目の動物）",
            "flamingo": "フラミンゴ",
            "electricfan": "扇風機",
            "weevil": "ゾウムシ",
            "trenchcoat": "トレンチコート",
            "robin": "鳥（ロビン／コマツグミ）",
            "persiancat": "ペルシャ猫",
            "lionfish": "ミノカサゴ",
            "tabby": "猫（トラ猫／ぶち猫）",
            "partridge": "鳥（ウズラ／ヤマウズラ）",
            "hartebeest": "ハーテビースト／シカレイヨウ",
            "three-toedsloth": "ナマケモノ（ミツユビナマケモノ）",
            "kingpenguin": "ペンギン（皇帝ペンギン）",
            "reflexcamera": "カメラ（レフレックス型・一眼レフ）",
            "langur": "猿（ラングール）",
            "seaslug": "ウミウシ",
            "briard": "犬（ブリアード）",
            "mongoose": "マングース",
            "skunk": "スカンク",
            "amphibian": "両生類",
            "sundial": "日時計",
            "scubadiver": "スキューバダイバー",
            "chiffonier": "西洋タンス",
            "tigerbeetle": "昆虫（ハンミョウ／オサムシ）",
            "nipple": "乳首",
            "cardigan": "カーディガン",
            "harvestman": "収穫者",
            "affenpinscher": "犬（アーフェンピンシャー）",
            "pineapple": "パイナップル",
            "snail": "かたつむり",
            "lycaenid": "蝶（シジミチョウ／アゲハ蝶）",
            "cock": "コック",
            "mortar": "モルタル",
            "coyote": "コヨーテ",
            "potpie": "ポットパイ（料理）",
            "footballhelmet": "footballhelmet",
            "custardapple": "果物（カスタードアップル／バンレイシ）",
            "toilettissue": "トイレットペーパー",
            "triumphalarch": "凱旋門",
            "mashedpotato": "マッシュポテト",
            "americanstaffordshireterrier": "犬（アメリカン・スタッフォードシャー・テリア）",
            "echidna": "動物（ハリモグラ／ハリネズミ）",
            "squirrelmonkey": "猿（リスザル）",
            "siamang": "猿（フクロテナガザル／テナガザル）",
            "bathtowel": "バスタオル",
            "waterouzel": "鳥（カワガラス）",
            "dungenesscrab": "カニ（ダンジネスクラブ）",
            "vinesnake": "ヘビ（ムチヘビ）",
            "bagel": "ベーグル",
            "silkyterrier": "犬（シルキー・テリア）",
            "vault": "大型金庫",
            "europeangallinule": "鳥（セイケイ）",
            "mexicanhairless": "犬（メキシカンヘアレス）",
            "miniaturepoodle": "犬（ミニチュア・プードル）",
            "cairn": "石積み",
            "sulphur-crestedcockatoo": "鳥（オウム目／キバタン）",
            "limpkin": "鳥（ツル目／ツルモドキ）",
            "wildboar": "イノシシ",
            "golfcart": "ゴルフカート",
            "toypoodle": "犬（トイプードル）",
            "bookcase": "書棚",
            "espresso": "エスプレッソ",
            "patas": "猿（パタスモンキー）",
            "howlermonkey": "猿（ホエザル）",
            "greenhouse": "温室",
            "goblet": "ゴブレット",
            "carbonara": "カルボナーラ",
            "potterswheel": "ろくろ",
            "goose": "ガチョウ",
            "orange": "オレンジ",
            "greenlizard": "トカゲ（ミドリカナヘビ）",
            "obelisk": "オベリスク（エジプトの記念碑）",
            "megalith": "巨石",
            "chimpanzee": "チンパンジー",
            "hornbill": "鳥（サイチョウ）",
            "espressomaker": "エスプレッソメーカー",
            "beerglass": "ビアグラス",
            "titi": "猿（ティティ）",
            "radiotelescope": "電波望遠鏡",
            "standardschnauzer": "犬（スタンダード・シュナウザー）r",
            "acorn": "どんぐり",
            "agaric": "寒天",
            "altar": "祭壇",
            "leafbeetle": "昆虫（ハムシ）",
            "norwegianelkhound": "犬（ノルウェジアン・エルクハウンド）",
            "guacamole": "料理（グアカモーレ／サルサ）",
            "kerryblueterrier": "犬（ケリー・ブルー・テリア）",
            "waterbuffalo": "水牛",
            "bluetick": "犬（ブルーティック・クーンハウンド）",
            "bouvierdesflandres": "犬（ブービエ・デ・フランダース）",
            "sulphurbutterfly": "蝶（ワタリオオキチョウ）",
            "patio": "パティオ（中庭・裏庭）",
            "miniaturepinscher": "犬（ミニチュア・ピンシャー）",
            "parkingmeter": "パーキングメーター",
            "red-breastedmerganser": "鳥（カモ目カモ科／ウミアイサ）",
            "manholecover": "マンホールのふた",
            "leonberg": "犬（レオンベルぐ）",
            "tibetanterrier": "犬（チベタン・テリア）",
            "angora": "猫（アンゴラ）／アンゴラウサギ",
            "tigershark": "イタチザメ",
            "teddy": "テディ（テディベア）",
            "leopard": "ヒョウ",
            "soapdispenser": "液体石鹸容器／ソープディスペンサー",
            "goldfish": "金魚",
            "red-backedsandpiper": "鳥（シギ）",
            "pekinese": "犬（ペキニーズ）",
            "rockbeauty": "魚（ロックビューティ／熱帯魚）",
            "jeep": "ジープ",
            "trilobite": "三葉虫",
            "greenmamba": "ヘビ（マンバ／グリーンマンバ）",
            "tibetanmastiff": "犬（チベタン・マスティフ）",
            "teapot": "ティーポット",
            "slidingdoor": "引き戸",
            "organ": "オルガン",
            "boxer": "犬（ボクサー）",
            "cashmachine": "ａｔｍ／キャッシュマシン",
            "coffeepot": "コーヒーポット",
            "plasticbag": "ビニール袋",
            "frenchloaf": "フレンチローフ／フランスパン",
            "mushroom": "キノコ（マッシュルーム）",
            "tailedfrog": "カエル（テイルドフロッグ）",
            "sealion": "アシカ",
            "artichoke": "アーティチョーク",
            "proboscismonkey": "猿（テングザル）",
            "greyfox": "キツネ（灰色狐）",
            "entertainmentcenter": "劇場・娯楽施設・音響付テレビ台（家具）",
            "marmoset": "猿（マーモセット）",
            "tigercat": "猫（ジャガー猫／トラ猫）",
            "gibbon": "猿（テナガザル）",
            "wallaby": "ワラビー",
            "lesserpanda": "レッサーパンダ",
            "platypus": "カモノハシ",
            "earthstar": "キノコ（アーススター）",
            "madagascarcat": "マダガスカルキャット",
            "pitcher": "ピッチャー",
            "windowscreen": "ウインドウスクリーン",
            "guenon": "猿（グエノン）",
            "ostrich": "ダチョウ",
            "diaper": "おむつ",
            "cauliflower": "カリフラワー",
            "indri": "猿（インドリ）",
            "hermitcrab": "ヤドカリ",
            "standardpoodle": "犬（プードル）",
            "spidermonkey": "猿（クモザル）",
            "americanblackbear": "クマ（アメリカグマ）",
            "ram": "ram（メモリ）",
            "frenchbulldog": "犬（フレンチ・ブルドッグ）",
            "mixingbowl": "ミキシングボウル",
            "sorrel": "植物（ソレル）",
            "saltshaker": "ソルトシェイカー",
            "soupbowl": "スープボール",
            "snowleopard": "ユキヒョウ",
            "gyromitra": "キノコ（ジャイロミトラ／アミガサタケ）",
            "washbasin": "洗面台",
            "kitfox": "キツネ（キットギツネ）",
            "modelt": "ｔ型自動車／クラッシックカー",
            "bee": "蜂",
            "chinacabinet": "家具（チャイナキャビネット）",
            "chamberednautilus": "オウム貝",
            "dough": "生地",
            "milkcan": "ミルク缶",
            "bellpepper": "ピーマン／パプリカ",
            "zucchini": "ズッキーニ",
            "tusker": "タスカー／ビール",
            "otter": "カワウソ",
            "africancrocodile": "アフリカのワニ",
            "fly": "昆虫（ハエ）",
            "cougar": "動物（クーガー）",
            "greatwhiteshark": "サメ（グレート・ホワイト・シャーク）",
            "geyser": "間欠泉",
            "icebear": "クマ（シロクマ／北極グマ）",
            "planetarium": "プラネタリウム",
            "bullmastiff": "犬（ブルマスティフ）",
            "lynx": "動物（リンクス／オオヤマネコ）",
            "fiddlercrab": "カニ（シオマネキ）",
            "lemon": "レモン",
            "turnstile": "回転木戸",
            "ringlet": "リングレット／長い巻毛",
            "peacock": "孔雀",
            "chiton": "生物（キトン／ヒザラガイ）",
            "terrapin": "カメ（テラピン）",
            "grille": "グリル",
            "colobus": "猿（コロブス）",
            "baboon": "ヒヒ",
            "fig": "イチジク",
            "tench": "魚（テンチ）",
            "mosquitonet": "蚊帳",
            "shih-tzu": "犬（シーズー）",
            "cardoon": "植物（カルドーン）",
            "woodrabbit": "木彫りのウサギ",
            "braincoral": "脳みたいな形のサンゴ",
            "indianelephant": "インド象",
            "axolotl": "サンショウウオ（メキシコオオサンショウウオ）",
            "giantpanda": "パンダ",
            "fountain": "噴水",
            "windowshade": "日よけ",
            "dhole": "犬（ドール）",
            "lhasa": "ラサ市／チベット",
            "chow": "犬（チャウ）",
            "policevan": "自動車（バン・警察用バン）",
            "africanhuntingdog": "犬（アフリカンハウンティングドッグ）",
            "spaghettisquash": "野菜（キンシウリ）",
            "killerwhale": "シャチ",
            "bighorn": "動物（ビッグホーン／ウシ科）",
            "whiskeyjug": "ウィスキー瓶（瀬戸物）",
            "tiger": "虎",
            "redwolf": "レッドウルフ",
            "dowitcher": "鳥（シギ科オオハシシギ）",
            "jack-o-lantern": "ハロウインのお化け（鬼火）",
            "rainbarrel": "雨水をためる樽",
            "butternutsquash": "野菜（バターナット・スクウォッシュカボチャ）",
            "treefrog": "アマガエル",
            "greensnake": "ヘビ（グリーンスネーク）",
            "hyena": "ハイエナ",
            "wombat": "ウォンバット",
            "crockpot": "電気なべ",
            "pay-phone": "公衆電話",
            "bison": "バイソン",
            "stingray": "アカエイ",
            "daisy": "花（デイジー）",
            "gorilla": "ゴリラ",
            "jackfruit": "果物（ジャックフルーツ）",
            "consomme": "コンソメスープ",
            "anemonefish": "魚（クマノミ）",
            "jaguar": "ジャガー",
            "leatherbackturtle": "カメ（ウミガメ）",
            "drake": "鳥（ドレイク／カモ属）",
            "slothbear": "ナマケモノ",
            "jellyfish": "クラゲ",
            "bustard": "鳥（ノガン／バスタード）",
            "lion": "ライオン",
            "hotpot": "火鍋",
            "komododragon": "コモドドラゴン",
            "bolete": "キノコ（ボレテ）",
            "petridish": "シャーレ（ペトリ皿）",
            "mudturtle": "カメ（泥カメ）",
            "cheeseburger": "チーズバーガー",
            "warthog": "イボイノシシ",
            "cicada": "蝉",
            "yellowladysslipper": "植物（イエローレディスリッパ）",
            "acornsquash": "どんぐり",
            "puffer": "フグ",
            "zebra": "シマウマ",
            "boxturtle": "カメ（ハコガメ）",
            "bullfrog": "ウシガエル",
            "loggerhead": "カメ（アカウミガメ）",
            "seaanemone": "イソギンチャク",
            "dungbeetle": "糞虫",
            "cabbagebutterfly": "蝶（モンシロチョウ）",
            "americancoot": "鳥（アメリカオオバン）",
            "dugong": "ジュゴン",
            "africanelephant": "アフリカ象",
            "koala": "コアラ",
            "hippopotamus": "カバ",
            "background": "背景",
        }

    def convert(self, english_name):
        replaced_str = re.sub(r"[\'_\s]", "", english_name.lower(), count=0)
        if replaced_str in self.jdic:
            return self.jdic[replaced_str]
        else:
            return "一致するキーがありません"


CLS_JP = Ilsvrc2012Japanese()

# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam/utils/
# image.py
# ==============================================
import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor


def preprocess_image(img: np.ndarray, mean=None, std=None) -> torch.Tensor:
    if std is None:
        std = [0.5, 0.5, 0.5]
    if mean is None:
        mean = [0.5, 0.5, 0.5]

    preprocessing = Compose([ToTensor(), Normalize(mean=mean, std=std)])

    return preprocessing(img.copy()).unsqueeze(0)


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def show_cam_on_image(
    img: np.ndarray,
    mask: np.ndarray,
    use_rgb: bool = False,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam
# guided_backprop.py
# ==============================================
import numpy as np
import torch
from torch.autograd import Function


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(
            torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask
        )
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input_img.size()).type_as(input_img),
            torch.addcmul(
                torch.zeros(input_img.size()).type_as(input_img),
                grad_output,
                positive_mask_1,
            ),
            positive_mask_2,
        )
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == "ReLU":
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]
        output = output.transpose((1, 2, 0))
        return output


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam
# activations_and_gradients.py
# ==============================================
class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam
# base_cam.py
# ==============================================
import cv2
import numpy as np
import torch

# from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


class BaseCAM:
    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layer, reshape_transform
        )

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self, input_tensor, target_category, activations, grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        return output[:, target_category]

    def get_cam_image(self, input_tensor, target_category, activations, grads):
        weights = self.get_cam_weights(
            input_tensor, target_category, activations, grads
        )
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        return cam

    def __call__(self, input_tensor, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        activations = (
            self.activations_and_grads.activations[-1].cpu().data.numpy()[0, :]
        )
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()[0, :]
        cam = self.get_cam_image(input_tensor, target_category, activations, grads)

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_tensor.shape[2:][::-1])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam
# grad_cam.py
# ==============================================
import cv2
import numpy as np
import torch

# from pytorch_grad_cam.base_cam import BaseCAM


class GradCAM(BaseCAM):
    """GradCAM:	Weight the 2D activations by the average gradient.
    平均勾配で2Dアクティベーションに重みを付け
    """

    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        super(GradCAM, self).__init__(model, target_layer, use_cuda, reshape_transform)

    def get_cam_weights(self, input_tensor, target_category, activations, grads):
        return np.mean(grads, axis=(1, 2))


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam
# grad_cam_plusplus.py
# ==============================================
import cv2
import numpy as np
import torch

# from pytorch_grad_cam.base_cam import BaseCAM


class GradCAMPlusPlus(BaseCAM):
    """GradCAM++: Like GradCAM but uses second order gradients.
    GradCAMと同様ですが、2次グラデーションを使用
    """

    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        super(GradCAMPlusPlus, self).__init__(
            model, target_layer, use_cuda, reshape_transform
        )

    def get_cam_weights(self, input_tensor, target_category, activations, grads):
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(1, 2))
        eps = 0.000001
        aij = grads_power_2 / (
            2 * grads_power_2 + sum_activations[:, None, None] * grads_power_3 + eps
        )

        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(1, 2))
        return weights


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam
# xgrad_cam.py
# ==============================================
import cv2
import numpy as np
import torch

# from pytorch_grad_cam.base_cam import BaseCAM


class XGradCAM(BaseCAM):
    """XGradCAM: Like GradCAM but scale the gradients by the normalized activations.
    GradCAMと同様ですが、正規化されたアクティベーションによってグラデーションをスケーリング
    """

    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        super(XGradCAM, self).__init__(model, target_layer, use_cuda, reshape_transform)

    def get_cam_weights(self, input_tensor, target_category, activations, grads):
        sum_activations = np.sum(activations, axis=(1, 2))
        eps = 1e-7
        weights = grads * activations / (sum_activations[:, None, None] + eps)
        weights = weights.sum(axis=(1, 2))
        return weights


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam
# ablation_cam.py
# ==============================================
import cv2
import numpy as np
import torch

# from pytorch_grad_cam.base_cam import BaseCAM


class AblationLayer(torch.nn.Module):
    def __init__(self, layer, reshape_transform, indices):
        super(AblationLayer, self).__init__()

        self.layer = layer
        self.reshape_transform = reshape_transform
        # The channels to zero out:
        self.indices = indices

    def forward(self, x):
        self.__call__(x)

    def __call__(self, x):
        output = self.layer(x)

        # Hack to work with ViT,
        # Since the activation channels are last and not first like in CNNs
        # Probably should remove it?
        if self.reshape_transform is not None:
            output = output.transpose(1, 2)

        for i in range(output.size(0)):

            # Commonly the minimum activation will be 0,
            # And then it makes sense to zero it out.
            # However depending on the architecture,
            # If the values can be negative, we use very negative values
            # to perform the ablation, deviating from the paper.
            if torch.min(output) == 0:
                output[i, self.indices[i], :] = 0
            else:
                ABLATION_VALUE = 1e5
                output[i, self.indices[i], :] = torch.min(output) - ABLATION_VALUE

        if self.reshape_transform is not None:
            output = output.transpose(2, 1)

        return output


def replace_layer_recursive(model, old_layer, new_layer):
    for name, layer in model._modules.items():
        if layer == old_layer:
            model._modules[name] = new_layer
            return True
        elif replace_layer_recursive(layer, old_layer, new_layer):
            return True
    return False


class AblationCAM(BaseCAM):
    """AblationCAM: Zero out activations and measure how the output drops (this repository includes a fast batched implementation)
    アクティベーションをゼロにし、出力がどのように低下​​するかを測定します（このリポジトリには高速バッチ実装が含まれています）
    """

    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        super(AblationCAM, self).__init__(
            model, target_layer, use_cuda, reshape_transform
        )

    def get_cam_weights(self, input_tensor, target_category, activations, grads):
        with torch.no_grad():
            original_score = self.model(input_tensor)[0, target_category].cpu().numpy()

        ablation_layer = AblationLayer(
            self.target_layer, self.reshape_transform, indices=[]
        )
        replace_layer_recursive(self.model, self.target_layer, ablation_layer)

        weights = []

        if hasattr(self, "batch_size"):
            BATCH_SIZE = self.batch_size
        else:
            BATCH_SIZE = 32

        with torch.no_grad():
            batch_tensor = input_tensor.repeat(BATCH_SIZE, 1, 1, 1)
            for i in range(0, activations.shape[0], BATCH_SIZE):
                ablation_layer.indices = list(range(i, i + BATCH_SIZE))

                if i + BATCH_SIZE > activations.shape[0]:
                    keep = i + BATCH_SIZE - activations.shape[0] - 1
                    batch_tensor = batch_tensor[:keep]
                    ablation_layer.indices = ablation_layer.indices[:keep]
                weights.extend(
                    self.model(batch_tensor)[:, target_category].cpu().numpy()
                )

        weights = np.float32(weights)
        weights = (original_score - weights) / original_score

        # replace the model back to the original state
        replace_layer_recursive(self.model, ablation_layer, self.target_layer)
        return weights


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam
# score_cam.py
# ==============================================
import cv2
import numpy as np
import torch

# from pytorch_grad_cam.base_cam import BaseCAM


class ScoreCAM(BaseCAM):
    """ScoreCAM: Perbutate the image by the scaled activations and measure how the output drops.
    スケーリングされたアクティベーションによって画像にパービュートし、出力がどのように低下​​するかを測定
    """

    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        super(ScoreCAM, self).__init__(
            model, target_layer, use_cuda, reshape_transform=reshape_transform
        )

    def get_cam_weights(self, input_tensor, target_category, activations, grads):
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(size=input_tensor.shape[2:])
            activation_tensor = torch.from_numpy(activations).unsqueeze(0)
            if self.cuda:
                activation_tensor = activation_tensor.cuda()

            upsampled = upsample(activation_tensor)
            upsampled = upsampled[
                0,
            ]

            maxs = upsampled.view(upsampled.size(0), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0), -1).min(dim=-1)[0]
            maxs, mins = maxs[:, None, None], mins[:, None, None]
            upsampled = (upsampled - mins) / (maxs - mins)

            input_tensors = input_tensor * upsampled[:, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            for i in range(0, input_tensors.size(0), BATCH_SIZE):
                batch = input_tensors[i : i + BATCH_SIZE, :]
                outputs = self.model(batch).cpu().numpy()[:, target_category]
                scores.append(outputs)
            scores = torch.from_numpy(np.concatenate(scores))
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam
# eigen_cam.py
# ==============================================
import cv2
import cv2
import numpy as np
import torch

# from pytorch_grad_cam.base_cam import BaseCAM

# https://arxiv.org/abs/2008.00299
class EigenCAM(BaseCAM):
    """EigenCAM:Takes the first principle component of the 2D Activations (no class discrimination, but seems to give great results).
    2Dアクティベーションの最初の主成分を取ります（クラスの識別はありませんが、素晴らしい結果が得られるようです）
    """

    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        super(EigenCAM, self).__init__(model, target_layer, use_cuda, reshape_transform)

    def get_cam_image(self, input_tensor, target_category, activations, grads):
        reshaped_activations = (
            (activations).reshape(activations.shape[0], -1).transpose()
        )
        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_activations = reshaped_activations - reshaped_activations.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])
        return projection


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master
# cam.py
# ==============================================
import argparse
import cv2
import numpy as np
import torch
from torchvision import models

# from pytorch_grad_cam import (GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,)
# from pytorch_grad_cam import GuidedBackpropReLUModel
# from pytorch_grad_cam.utils.image import (show_cam_on_image, deprocess_image, preprocess_image,)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        default=False,
        help="Use NVIDIA GPU acceleration",
    )
    # parser.add_argument("--image-path", type=str, default="./image/dog.jpg", help="Input image path")
    parser.add_argument(
        "--method",
        type=str,
        default="gradcam",
        help="Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam",
    )
    parser.add_argument("--model", type=str, default="resnet18")  # 追加

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    # if args.use_cuda:
    #    print("Using GPU for acceleration")
    # else:
    #    print("Using CPU for computation")

    return args


def cam_main(args, model, target_layer, cv2_img, target_category=None):
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        # "eigengradcam": EigenGradCAM,
    }

    # models.resnet50(pretrained=True)

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # target_layer = model.layer4[-1]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    cam = methods[args.method](
        model=model, target_layer=target_layer, use_cuda=args.use_cuda
    )

    # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2_img[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(
        rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    # target_category = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    # gb = gb_model(input_tensor, target_category=target_category)

    # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)

    # cv2.imwrite(f"{args.method}_cam.jpg", cam_image)
    # cv2.imwrite(f"{args.method}_gb.jpg", gb)
    # cv2.imwrite(f"{args.method}_cam_gb.jpg", cam_gb)

    return cam_image


# ==============================================
# app.py
# ==============================================
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image


@st.cache(allow_output_mutation=True)
def load_model(model):
    if model == "resnet18":
        model = models.resnet18(pretrained=True)
        target_layer = model.layer4[-1]
    elif model == "resnet50":
        model = models.resnet50(pretrained=True)
        target_layer = model.layer4[-1]
    elif model == "resnet101":
        model = models.resnet101(pretrained=True)
        target_layer = model.layer4[-1]
    # model = models.mobilenet_v2(pretrained=True)
    return model, target_layer


def load_file_up_image(file_up, size=224):
    pillow_img = Image.open(file_up).convert("RGB")
    pillow_img = pillow_img.resize((size, size)) if size is not None else pillow_img
    cv2_img = pil2cv(pillow_img)
    return pillow_img, cv2_img


def pil2cv(pillow_img):
    """ PIL型 -> OpenCV型
    https://qiita.com/derodero24/items/f22c22b22451609908ee"""
    new_image = np.array(pillow_img, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def preprocessing_image(image_pil_array: "PIL.Image"):
    # https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose(
        [
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch_t = torch.unsqueeze(transform(image_pil_array), 0)
    return batch_t


def predict(args, pillow_img, cv2_img):
    batch_t = preprocessing_image(pillow_img)

    model, target_layer = load_model(args.model)
    model.eval()
    out = model(batch_t)

    with open("imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    top5 = [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

    top1_id = top5[0][0].split(",")[0]
    cam_image = cam_main(
        args, model, target_layer, cv2_img, target_category=int(top1_id)
    )
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)

    return top5, cam_image


def main():
    st.set_option("deprecation.showfileUploaderEncoding", False)
    st.title("Simple Image Classification App")
    st.write("")

    args = get_args()

    # ファイルupload
    file_up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    # サイドバー ラジオボタン
    st_model = st.sidebar.radio(
        "Select ImageNet Model",
        (
            "resnet18",
            "resnet50",
            # "resnet101",
        ),
    )
    args.__setattr__("model", st_model)

    # サイドバー ラジオボタン
    now_st_method = ""
    st_method = st.sidebar.radio(
        "Select Class Activation Map Method",
        (
            "gradcam",
            "gradcam++",
            "xgradcam",
            "ablationcam",
            "scorecam",
            "eigencam",
            # "eigengradcam",
        ),
    )
    args.__setattr__("method", st_method)  # args.method

    # サイドバー ラジオボタン
    st_is_jp_class = st.sidebar.radio("予測のクラス名を日本語にする", ("True", "False",))

    def run():
        pillow_img, cv2_img = load_file_up_image(file_up)

        st.image(
            pillow_img,
            caption="Uploaded Image. Resize (224, 224).",
            use_column_width=True,
        )

        st.write("")
        # st.write("Just a second...")
        labels, cam_image = predict(args, pillow_img, cv2_img)
        st.image(
            cam_image.transpose(0, 1, 2),
            caption="Class Activation Map Method: " + st_method,
            use_column_width=True,
        )

        # print out the top 5 prediction labels with scores
        # 1位だけにする
        for i in labels[:1]:
            class_id = int(i[0].split(", ")[0])
            class_name = i[0].split(", ")[1]
            if eval(st_is_jp_class):
                class_name = CLS_JP.convert(class_name)
            st.write(
                f"# Prediction: **{class_name}**"
            )  #  (imagenet id: {str(class_id)})
            st.write("## score: ", round(i[1], 1), " %")

        now_st_method = st_method

    if file_up is not None and now_st_method != st_method:
        run()
    else:
        img_url = "https://github.com/riron1206/image_classification_streamlit_app/blob/master/image/dog.jpg?raw=true"
        st.image(
            img_url,
            caption="Sample Image. Please download and upload.",
            use_column_width=True,
        )


if __name__ == "__main__":
    main()
