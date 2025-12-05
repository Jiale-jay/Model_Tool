import click
import torch
from models import DFAOITNet, DFAOITNetConv, DFAOITNetShaderVersion
from utils import load_weights_from_csharp, load_existing_weights
from training import generate_consistency_data, simple_fine_tune, test_rgba_consistency
import os
import subprocess
import onnx
from onnx import TensorShapeProto



@click.group()
def cli():

    """
    DFAOIT model toolset: training, fine-tuning, exporting ONNX,Reshape
    
    Command:
    1.python Main_cli_tool.py train --samples 20000 --output ModelName.pth
    
    2.python Main_cli_tool.py finetune --init Model_Name.pth --samples 10000 --output finetuned_ModelName.pth

    3.python Main_cli_tool.py export --pth Model_Name.pth --output Model_Name.onnx  

    4.python Main_cli_tool.py reshape --input model.onnx --height 1080 --width 1920 --output model_1080x1920.onnx

    5.python Main_cli_tool.py export-fp16 --pth DFAOITModel.pth --output DFAOITModel_fp16.onnx

    6.python Main_cli_tool.py compare-fp16 --pth default.pth --model_arch DFAOITNetConv
    """
    pass

@cli.command()
@click.option('--samples', default=15000, show_default=True, type=int, help='Number of training samples')
@click.option('--output', default='DFAOITModel.pth', show_default=True, type=str, help='Save model path')

def train(samples, output):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DFAOITNet().to(device)
    
    csharp_weights = """
    private static float[] weights1 = { -1.1959514617919922f, 0.008686563931405544f, -0.1937752366065979f, 1.0385195016860962f, 0.19693408906459808f, 0.2250373661518097f, 0.6747682690620422f, -1.4266048669815063f, 0.3059764504432678f, 0.399554044008255f, 0.27530235052108765f, 0.5219589471817017f, 0.08253791928291321f, -1.228529930114746f, -0.002411186695098877f, -1.5230942964553833f, 0.9490935802459717f, -0.5497636198997498f, -0.5618413090705872f, -0.0033106456976383924f, 0.561732292175293f, -0.12235775589942932f, 0.8593060970306396f, 0.19862844049930573f, -0.7808619737625122f, 0.14008311927318573f, -0.019419029355049133f, 0.07229571789503098f, 0.20560601353645325f, 0.5272841453552246f, -0.453464150428772f, 0.20026898384094238f, -0.32112884521484375f, -0.18358077108860016f, 1.9115184545516968f, -0.14861445128917694f, 0.4587515890598297f, -0.24115443229675293f, -0.019700361415743828f, 1.417148232460022f, -0.344126433134079f, -1.6651922464370728f, -2.0527474880218506f, -0.3778584599494934f, -0.19501537084579468f, -1.1878787279129028f, 0.2124522626399994f, 0.3017037808895111f, 0.2961932122707367f, 0.4879889488220215f, -0.5376386642456055f, -0.31061938405036926f, -1.1316797733306885f, -0.909258246421814f, -0.2944614589214325f, 0.09972604364156723f, -12.067819595336914f, 1.5719071626663208f, -0.3163011372089386f, 0.6028306484222412f, -0.0460175946354866f, -3.6968750953674316f, 0.43281474709510803f, 0.7987827658653259f, -0.747306227684021f, -0.2081216424703598f, 1.6365711688995361f, -0.4465257525444031f, 0.33296605944633484f, -0.29781994223594666f, -0.1783263087272644f, 0.25825634598731995f, 0.19828782975673676f, 0.6044086813926697f, 0.28318580985069275f, -1.072197437286377f, 0.008227660320699215f, -0.1253279745578766f, -0.0783214271068573f, 0.9278473854064941f, -0.8836387395858765f, -0.33477675914764404f, -0.3048165440559387f, -0.18421262502670288f, -0.43830668926239014f, -1.6739978790283203f, -1.9730682373046875f, -2.1032328605651855f, -18.26808738708496f, 0.30352431535720825f, -0.06819019466638565f, 0.12744948267936707f, -3.205310583114624f, -0.8645298480987549f, -2.3543710708618164f, -0.18044570088386536f, 0.8616771101951599f, 0.3256019055843353f, -0.7677901387214661f, 0.6860630512237549f, -2.311643123626709f, 0.8824611902236938f, -0.4228302538394928f, -0.055094894021749496f, 0.37132036685943604f, 0.15426534414291382f, 0.7204028964042664f, 0.00037511371192522347f, 0.06806614249944687f, 1.2826141119003296f, -0.4323919713497162f, 0.4851935803890228f, -0.5826048851013184f, 0.07026584446430206f, 1.0444015264511108f, 1.5576900243759155f, 0.7251351475715637f, 0.6116061806678772f, 0.797327995300293f, 1.0446865558624268f, -9.101639747619629f, -2.3920469284057617f, 0.03743177279829979f, -0.843485951423645f, -0.7569155097007751f, 1.1005421876907349f, 2.3064351081848145f, -0.2463560700416565f, -0.32280752062797546f, 0.020834308117628098f, -1.4779772758483887f, 0.038507129997015f, -0.0862230435013771f, -0.24627602100372314f, 0.3041973114013672f, 0.14891831576824188f, 0.21034608781337738f, 0.44557350873947144f, 0.2671230435371399f, 0.12094470858573914f, 0.8080748319625854f, -0.12898804247379303f, -0.09779750555753708f, -0.16937418282032013f, -0.22657562792301178f, 0.611360490322113f, 0.20182126760482788f, -0.04682571440935135f, -0.34075403213500977f, 0.40388306975364685f, 0.30762118101119995f, 0.10296429693698883f, 0.15936221182346344f, -0.010127464309334755f, 0.015392914414405823f, -0.07913041114807129f, -0.054541509598493576f, 0.34538736939430237f, -0.25479599833488464f, 0.6687596440315247f, -0.09726037085056305f, 0.098298579454422f, -1.1525148153305054f, -0.006606179289519787f, -0.2517842948436737f, -0.24290618300437927f, 0.6350741386413574f, -0.15681207180023193f, 0.5440434217453003f, -0.24486859142780304f, -0.1493503749370575f, -0.006185303907841444f, 0.962202250957489f, 0.01748655177652836f, 0.033148352056741714f, 0.1075744777917862f, 0.06570865958929062f, 0.35085809230804443f, 0.8330182433128357f, 0.03134474903345108f, 0.6706594228744507f, 0.8656180500984192f, -0.25917497277259827f, -0.015072023496031761f, 0.014841337688267231f, 0.08644524216651917f, -0.03954117372632027f, 0.002060219645500183f, -0.9966229200363159f, -0.20114032924175262f, 0.05233042687177658f, -0.35363703966140747f, -1.1465520858764648f, -0.09708844125270844f, -0.2554410398006439f, -0.157853364944458f, 0.501046359539032f, -11.565048217773438f, 0.016360998153686523f, -0.037242770195007324f, -0.19894124567508698f, -0.299430251121521f, -0.19109228253364563f, -0.13984794914722443f, -0.020829396322369576f, 0.0602877214550972f, 0.08991197496652603f, 0.029336605221033096f, 0.07054608315229416f, -0.17844608426094055f, 0.06280391663312912f, -0.046232156455516815f, -0.3195935785770416f, -0.24996818602085114f, -0.09213313460350037f, -0.12648040056228638f, 0.7795014381408691f, -0.19904878735542297f, 0.05011957511305809f, 0.07649651169776917f, 1.360342025756836f, -0.1383044570684433f, 0.41980209946632385f, -0.12338314205408096f, 0.42042773962020874f, -1.577932357788086f, 1.0078223943710327f, -1.0605205297470093f, 0.35287901759147644f, 0.9932248592376709f, 1.653694748878479f, -0.8026828765869141f, -0.7746771574020386f, 0.06780596822500229f, -4.55473518371582f, 0.45145905017852783f, 0.6953707337379456f, 1.0300620794296265f, -3.5986011028289795f, 0.5053223967552185f, -0.32716161012649536f, -2.3369805812835693f, 1.6742128133773804f, -0.012954497709870338f, 1.4992144107818604f, -0.4427145719528198f, 0.5302993059158325f, 0.009094552136957645f, -0.3936215937137604f, 0.13822458684444427f, -0.20050887763500214f, 0.7013229131698608f, 0.507688581943512f, -11.029073715209961f, 0.09855765849351883f, 1.7905081510543823f, 0.09672455489635468f, -0.4092945158481598f, 1.5038893222808838f, -1.0378752946853638f, 0.5929093956947327f, -0.1868642121553421f, 0.7708125114440918f, -0.09402211755514145f, 3.138631582260132f, 0.29342177510261536f, 0.5072694420814514f, -7.3871684074401855f, -0.7743351459503174f, 2.616819381713867f, 0.5649107694625854f, -0.5819246768951416f, 0.8473128080368042f, 0.8983940482139587f, -0.8343707919120789f, -0.44072848558425903f, -1.1694471836090088f, -0.49466848373413086f, 1.1135810613632202f, -3.7073304653167725f, -0.338094562292099f, 0.25775885581970215f, -0.36888587474823f, -3.449357271194458f, -18.542001724243164f, 2.04738712310791f, -1.3176251649856567f, 1.6326231956481934f, 1.7182526588439941f, 0.2149658501148224f, -1.0617996454238892f, 1.1388027667999268f, -7.008763313293457f, -4.764480113983154f, -2.8290886878967285f, 0.07904283702373505f, 1.0919829607009888f, -4.736726760864258f, 0.7009994983673096f, 0.6952314972877502f, -0.3030679225921631f, 2.4487380981445312f, 0.10241623222827911f, -0.3376392126083374f, 2.523329973220825f, 0.8281531929969788f, 2.6658360958099365f, 3.98511004447937f, 0.5961707234382629f, 0.8543791770935059f, 0.11942192912101746f, -0.1556708961725235f, -1.1582932472229004f, -3.1259171962738037f, -0.2020474076271057f, 0.061312925070524216f, -1.456930160522461f, -1.5318260192871094f, 1.546918272972107f, 1.4512051343917847f };
    private static float[] bias1 = { 0.94788074f, 1.1071887f, 0.17212379f, 0.96233034f, 0.41116345f, 0.29424584f, 0.2524292f, 0.56384075f, 0.17144501f, 0.70693594f, 0.60297567f, 0.3995347f, -0.11675993f, 0.6551702f, 0.8530541f, 0.8074819f, -0.14349526f, -0.02273662f, 0.22370361f, -0.15556079f, -0.10795841f, -0.04227992f, -0.2230586f, 0.57490826f, 0.35543177f, 0.30263418f, -0.01075008f, 0.57908285f, 0.35843596f, 0.50211096f, 0.17400633f, 0.00989959f };

    private static float[] weights2 = { 0.4855324625968933f, 0.4931071400642395f, 0.10380104184150696f, 1.3709137439727783f, -0.32564836740493774f, -1.9942959547042847f, 1.386703610420227f, 0.3648749589920044f, 2.0565879344940186f, 0.02649473398923874f, -0.682405412197113f, 0.19458186626434326f, -0.17608760297298431f, 0.7249542474746704f, 0.38143762946128845f, -0.060828015208244324f, -0.15595318377017975f, -0.7519590258598328f, 0.07108820974826813f, 0.7860511541366577f, -0.8302670121192932f, -0.22049321234226227f, 0.14070487022399902f, 0.2799070477485657f, -2.1001341342926025f, -0.057683296501636505f, -0.014779577031731606f, -0.7197736501693726f, 0.49455758929252625f, -0.10179200768470764f, 1.1618152856826782f, -0.24636442959308624f, 0.3151477575302124f, 0.24846771359443665f, 0.10003579407930374f, 0.4288254976272583f, -0.13821683824062347f, -0.38535216450691223f, 0.4654299318790436f, 0.02280261740088463f, 1.0234557390213013f, 0.020582543686032295f, 0.15840017795562744f, 0.10749263316392899f, 1.3036932945251465f, 1.1788554191589355f, 0.36588141322135925f, -0.31320664286613464f, -0.5017208456993103f, 0.9030942916870117f, 0.7830336093902588f, -0.25512924790382385f, 1.2962630987167358f, 0.9618062973022461f, 1.457244873046875f, -0.8943830728530884f, 0.32682719826698303f, 0.18003877997398376f, -0.39415791630744934f, 0.5483765006065369f, 0.36082711815834045f, -0.04673543572425842f, 0.7956789135932922f, -0.03845515102148056f, 0.5876906514167786f, 0.0006338664679788053f, 1.0915290117263794f, 0.6103436946868896f, 0.6506611108779907f, 0.5670680999755859f, -0.7777470946311951f, -0.16347284615039825f, 0.43014267086982727f, -0.2577018439769745f, -0.4195305109024048f, -0.30550628900527954f, -0.050949689000844955f, -1.3560826778411865f, 0.2832963466644287f, -0.38394439220428467f, -1.0235261917114258f, -0.17534871399402618f, -4.8392109870910645f, 0.16609767079353333f, 0.823218584060669f, -3.0942225456237793f, 1.2025632858276367f, 4.317955493927002f, 1.4691383838653564f, 0.16346687078475952f, -0.4719548523426056f, 0.5866063833236694f, 0.4686736762523651f, 0.7369234561920166f, 0.008164627477526665f, -0.17022749781608582f, 0.5724796056747437f, -0.03635172173380852f, -0.1545363813638687f, 1.039222240447998f, -0.3984861969947815f, 0.0296646561473608f, -0.2836281359195709f, 1.1882721185684204f, 0.1366232931613922f, -0.204912930727005f, 0.007424644660204649f, -1.0028709173202515f, -1.202796220779419f, -0.05829732492566109f, -1.7417320013046265f, 0.04341378062963486f, 0.23129233717918396f, -1.3527560234069824f, -0.42133307456970215f, 0.12211021035909653f, -0.6104382872581482f, -0.1868549883365631f, 0.40106314420700073f, -0.4845059812068939f, -0.0197924692183733f, 0.12606611847877502f, 0.5827474594116211f, 0.6700010299682617f, 0.05103679001331329f, 0.5389978885650635f, -0.23360837996006012f, 0.23208793997764587f, 0.67277991771698f, -1.7657382488250732f, 0.02171258069574833f, 0.08498626202344894f, 0.28049126267433167f, -0.7595258355140686f, 0.2776927053928375f, -0.35044965147972107f, 0.4565054476261139f, -0.15233522653579712f, 0.4872066378593445f, 1.300654649734497f, -1.8504332304000854f, 0.8463921546936035f, -0.2597326338291168f, 0.22363460063934326f, 0.010712376795709133f, -0.4714129567146301f, 0.35520726442337036f, 1.446079969406128f, 0.5455968976020813f, -1.3260340690612793f, 0.19403614103794098f, 0.049743350595235825f, 0.47353142499923706f, 0.12994804978370667f, -0.9549532532691956f, -0.015963705256581306f, -0.38273322582244873f, -0.9555284976959229f, -0.19458109140396118f, -0.36547940969467163f, -0.7875256538391113f, 0.3425721824169159f, -1.2423834800720215f, -0.4068931043148041f, 1.01790189743042f, 0.22240284085273743f, 0.3558140993118286f, 0.2660442292690277f, -0.5474269390106201f, -0.3615276515483856f, 1.0313496589660645f, 0.4277484714984894f, -0.45575040578842163f, 2.0379459857940674f, 0.7719808220863342f, -0.2623959183692932f, -0.11566110700368881f, 0.15549589693546295f, 0.8930090665817261f, 0.9349038600921631f, 0.7597079873085022f, 0.05463235452771187f, 0.8461006879806519f, 0.12101807445287704f, -0.6079994440078735f, 0.004490119405090809f, 1.1005821228027344f, -0.27633994817733765f, 0.14224164187908173f, 0.34271568059921265f, 0.16557490825653076f, -0.3457490801811218f, 0.39591705799102783f, -0.2504904568195343f, -0.031493984162807465f, -0.18310216069221497f, 0.7079054117202759f, -0.1883230060338974f, 0.34667375683784485f, -0.07623184472322464f, 0.2013239711523056f, -0.11317132413387299f, -0.11856567114591599f, -0.015944575890898705f, 3.6823577880859375f, 0.18516729772090912f, 2.315159797668457f, -0.2989019751548767f, 0.5496065020561218f, 0.7007465958595276f, 0.17294569313526154f, -0.7926169037818909f, 0.6532262563705444f, 1.5787855386734009f, 0.2006942331790924f, -0.18958544731140137f, -0.28471508622169495f, -0.18025937676429749f, 0.378176212310791f, -0.3235878646373749f, 2.6459484100341797f, -0.8277072310447693f, 0.23949141800403595f, 0.18704336881637573f, -0.3619232475757599f, -0.583285391330719f, -0.9040329456329346f, -0.4538056254386902f, 1.512380838394165f, 0.31583330035209656f, 0.9610786437988281f, -0.08723882585763931f, 1.027336597442627f, 0.01928548514842987f, -0.2750968933105469f, -0.46684974431991577f, 0.1602950543165207f, -1.5791192054748535f, 0.5341970920562744f, -0.39536944031715393f, 0.1799522489309311f, 0.923843264579773f, 0.21015281975269318f, -0.2897454500198364f, 0.39070063829421997f, 0.41485702991485596f, -0.31079035997390747f, -0.0005115203093737364f, -0.5140342116355896f, -0.1640966832637787f, 1.1543800830841064f, -0.518275260925293f, -1.577041506767273f, -0.013632168993353844f, -0.4515509009361267f, 0.06432336568832397f, 0.09957201778888702f, -0.09353253990411758f, 0.12263648957014084f, -0.690255880355835f, -0.19034689664840698f, 0.8746852874755859f, -0.9009305834770203f, -0.313703328371048f, 0.5521869659423828f, -0.19129839539527893f, 1.2473962306976318f, 0.061218325048685074f, 0.08761314302682877f, 0.654431164264679f, -0.14619199931621552f, -0.33451715111732483f, 0.07974260300397873f, -0.37096571922302246f, -0.06174737960100174f, -0.8915731906890869f, -0.5108177661895752f, 1.1457362174987793f, -0.493486613035202f, 0.07922235876321793f, 0.22418564558029175f, 0.19242951273918152f, 0.3291654884815216f, 0.38492754101753235f, -0.8371371626853943f, -0.4161410927772522f, -0.8019778728485107f, 0.21948641538619995f, -0.7994012236595154f, -1.8920094966888428f, 0.6720749139785767f, -0.17469771206378937f, -0.3729900121688843f, 0.023695070296525955f, -0.021520232781767845f, -0.026918785646557808f, 0.44524917006492615f, -0.1861088126897812f, 0.978980302810669f, -0.5237755179405212f, -2.601365804672241f, -0.5347180962562561f, -1.005389928817749f, -0.08774088323116302f, 0.6610136032104492f, -0.33315953612327576f, -0.5808750987052917f, 1.0279815196990967f, -0.6386284828186035f, -1.6286004781723022f, -0.024559464305639267f, 1.7753922939300537f, -0.1544920802116394f, -0.18138431012630463f, -0.347635954618454f, -0.8881171941757202f, -0.17921799421310425f, 1.1944081783294678f, 0.06857842206954956f, -0.12952955067157745f, -0.3381940424442291f, -1.0793044567108154f, 0.37167641520500183f, 0.32332950830459595f, 0.008518817834556103f, 0.2817102074623108f, -0.4100678265094757f, -0.21096022427082062f, -0.9516218900680542f, -0.17738547921180725f, 0.28829070925712585f, 0.32393237948417664f, 0.14444445073604584f, -0.7172508239746094f, -0.06652387976646423f, 0.13653846085071564f, -0.4724299907684326f, 3.959125280380249f, -0.595298171043396f, -0.11153925955295563f, 0.05432121083140373f, 0.05404157564043999f, 0.31248193979263306f, -0.45154955983161926f, -0.5873998403549194f, 0.32329025864601135f, -1.105285406112671f, 0.47186627984046936f, -0.7073745727539062f, 0.04058119282126427f, -0.25135353207588196f, 0.06204336881637573f, -0.09862303733825684f, -0.16838370263576508f, -0.6563517451286316f, -0.33635249733924866f, 0.7841166257858276f, -0.2100493609905243f, 0.23235458135604858f, 0.6351258158683777f, 0.25170159339904785f, -0.08564119040966034f, 1.1780649423599243f, -0.8901026248931885f, -0.3950941264629364f, 0.43442872166633606f, -0.34140560030937195f, -0.20018291473388672f, -0.9007940292358398f, -0.7312206029891968f, -1.2381744384765625f, 0.3201162815093994f, 0.49314022064208984f, -0.41883111000061035f, 1.3737378120422363f, -0.07130175083875656f, 0.004479328170418739f, -0.10741174221038818f, -0.40654563903808594f, 0.28353869915008545f, 0.11062107980251312f, -0.11100152879953384f, -0.13095390796661377f, -0.17056450247764587f, 1.441879153251648f, 0.4924049377441406f, 3.0137619972229004f, 1.6133272647857666f, -0.667236328125f, 0.12797600030899048f, -0.5594787001609802f, 1.2254446744918823f, -0.20554421842098236f, -0.18752239644527435f, -0.02557636797428131f, 1.4591253995895386f, -1.95885169506073f, 1.8351293802261353f, 1.1069278717041016f, -0.23766392469406128f, 0.1360471546649933f, 0.2355700135231018f, 0.1110483780503273f, 0.8886839151382446f, 0.15384434163570404f, -0.2800663113594055f, -0.3006570041179657f, 0.699007511138916f, -0.4770301878452301f, 0.0030932750087231398f, 0.3729117512702942f, 0.5146426558494568f, -0.08564124256372452f, 1.5841659307479858f, -0.23869852721691132f, -0.17625218629837036f, 0.06073814630508423f, 0.3504723012447357f, 0.20597414672374725f, 0.1500907689332962f, -0.2402230054140091f, 0.31028828024864197f, -0.2294100522994995f, -0.22374005615711212f, -0.3091212511062622f, 0.1410476118326187f, 0.028931617736816406f, 0.2358057200908661f, 0.10067448019981384f, 0.09879446774721146f, 0.27338817715644836f, 0.21246185898780823f, -0.24570997059345245f, -0.02542683482170105f, 0.285625159740448f, 0.7275424003601074f, -0.18951316177845f, -0.2462969720363617f, 0.04082686826586723f, -0.5502161383628845f, 0.813132107257843f, 0.06461475044488907f, 0.1457694172859192f, 0.4810294210910797f, -0.5208631753921509f, 1.0722259283065796f, -1.654968500137329f, 0.1312693953514099f, 0.4604164958000183f, -0.2767646908760071f, 6.574335098266602f, 4.611900806427002f, -0.5481435060501099f, 0.12950026988983154f, 1.8382775783538818f, 0.19294202327728271f, 0.26653245091438293f, -0.30541419982910156f, 0.9513653516769409f, 0.7260265946388245f, -0.18213582038879395f, -0.2617158889770508f, 0.21246477961540222f, -0.10989498347043991f, 1.8427926301956177f, 0.3394126892089844f, 0.9705246090888977f, -0.2986394762992859f, 1.8211095333099365f, 1.890419840812683f, 1.187295913696289f, 0.15032079815864563f, -0.35558509826660156f, 0.038662560284137726f, 1.474493384361267f, 0.34445011615753174f, 0.5976400375366211f, -1.4448442459106445f, 0.4462164342403412f, 0.210927814245224f, -0.06704511493444443f, 0.8425297737121582f, 0.7273149490356445f, 0.19566695392131805f, -0.04238157719373703f, 0.1792774349451065f, -0.8822489380836487f, -0.8191877007484436f, 1.3391218185424805f, -0.31876859068870544f, 1.0471078157424927f, 0.42806822061538696f, 0.02055804617702961f, 0.8125109672546387f, -1.0384373664855957f, 0.06425516307353973f, -0.29669493436813354f, 0.33801916241645813f, 0.5476243495941162f, 1.0462514162063599f, -0.7626770734786987f, -0.11118371039628983f, -0.3406493067741394f, -1.163106918334961f, -0.6331011056900024f, 0.0976642593741417f, 0.656583845615387f, 1.0642417669296265f, -0.5848557353019714f, 0.6343390345573425f, -0.9152898192405701f, -0.2263033539056778f };
    private static float[] bias2 = { 1.0024987f, 0.7019234f, -0.31018218f, -0.2966683f, 0.57583773f, 0.38118008f, 0.03561725f, 1.030341f, 0.00273793f, -0.092371f, -0.50508595f, -0.23851559f, 1.160991f, -0.05685527f, 0.00209139f, -0.08914163f };

    private static float[] weights3 = { -0.1631498485803604f, 0.5885962843894958f, -0.45753711462020874f, -0.29011666774749756f, -1.7414355278015137f, -0.254714697599411f, -0.3687281906604767f, -0.42367491126060486f, 0.4668368399143219f, 0.5017827749252319f, -0.35678642988204956f, -0.7905645966529846f, -0.5253183841705322f, 0.17899642884731293f, -0.08671782165765762f, 0.05534752458333969f, -0.5685427784919739f, -0.6783115863800049f, -0.24065136909484863f, -0.5730992555618286f, 0.0858161449432373f, -0.13739177584648132f, -0.06401057541370392f, -0.6701674461364746f, -0.17737291753292084f, -0.17955224215984344f, -0.19217659533023834f, 0.4204261302947998f, 0.5289128422737122f, 0.3415016531944275f, 0.4365029036998749f, 0.2399706393480301f, 0.4704277217388153f, -0.1925477236509323f, 0.21371600031852722f, 0.3034309446811676f, -1.1344658136367798f, -0.982851505279541f, -0.9596925377845764f, -0.32806965708732605f, -0.21451649069786072f, -0.20496052503585815f, -0.43247565627098083f, -0.06669460237026215f, -0.03655197471380234f, 0.06050315126776695f, 0.32564544677734375f, 0.19158335030078888f };
    private static float[] bias3 = { -0.7872798f, 0.04166554f, -0.61643946f };
    """
    if len(csharp_weights.strip()) < 100:
        print("First,copy weights")
        return
    try:
        w1, b1, w2, b2, w3, b3 = load_weights_from_csharp(csharp_weights)
        load_existing_weights(model, w1, b1, w2, b2, w3, b3, device=device)
        print("Success!")
    except Exception as e:
        print(f"Fail: {e}")
        return
    all_inputs, all_targets = generate_consistency_data(model, num_samples=samples)
    train_size = int(0.8 * len(all_inputs))
    train_inputs = all_inputs[:train_size]
    train_targets = all_targets[:train_size]
    val_inputs = all_inputs[train_size:]
    val_targets = all_targets[train_size:]
    print(f"Train set: {len(train_inputs)}，Validation set: {len(val_inputs)}")
    history = simple_fine_tune(model, train_inputs, train_targets, val_inputs, val_targets)
    torch.save(model.state_dict(), output)
    
    print("Success，Save model to", output)

@cli.command()
@click.option('--init', required=True, type=click.Path(exists=True), help='Initial Weight PTH File')
@click.option('--samples', default=20000, show_default=True, type=int, help='Fine-tune the number of training samples')
@click.option('--output', default='finetuned_DFAOITModel.pth', show_default=True, type=str, help='save path')
def finetune(init, samples, output):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DFAOITNet().to(device)
    model.load_state_dict(torch.load(init, map_location=device))
    print(f"The initial weight file has been loaded.: {init}")
    all_inputs, all_targets = generate_consistency_data(model, num_samples=samples)
    train_size = int(0.8 * len(all_inputs))
    train_inputs = all_inputs[:train_size]
    train_targets = all_targets[:train_size]
    val_inputs = all_inputs[train_size:]
    val_targets = all_targets[train_size:]
    print(f"Train set: {len(train_inputs)}，Validation set: {len(val_inputs)}")
    history = simple_fine_tune(model, train_inputs, train_targets, val_inputs, val_targets)
    torch.save(model.state_dict(), output)
    print("Fine-tuning completed，save to", output)

@cli.command()
@click.option('--pth', type=click.Path(exists=True), default='DFAOITModel.pth', help='PyTorch 权重路径')
@click.option('--output', default='DFAOITModel.onnx', show_default=True, type=str, help='ONNX 输出路径')
@click.option('--dummy_h', default=16, show_default=True, type=int, help='导出用占位高度（仅构图用，实际推理支持动态）')
@click.option('--dummy_w', default=16, show_default=True, type=int, help='导出用占位宽度（仅构图用，实际推理支持动态）')
@click.option("--model_arch", type=click.Choice(["DFAOITNet", "DFAOITNetConv"]), default="DFAOITNet", help="Chose the model architecture to export")
@click.option('--use_dynamic_axes', is_flag=True, default=False, help = "Allow dynamic model input size")
def export(**kwargs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_class = None
    if kwargs["model_arch"] == "DFAOITNet":
        model_class = DFAOITNet
        input_shape = (1, kwargs['dummy_h'], kwargs['dummy_w'], 10)
    elif kwargs["model_arch"] == "DFAOITNetConv":
        model_class = DFAOITNetConv
        input_shape = (1, 10, kwargs['dummy_h'], kwargs['dummy_w'])
    else:
        raise NotImplementedError("Model architecture '%s' is no supported" % kwargs["model_arch"])

    model = model_class().to(device)
    
    try:
        state_dict = torch.load(kwargs['pth'], map_location=device)
        
        if kwargs["model_arch"] == "DFAOITNetConv":
            if state_dict['layer1.weight'].shape != model.state_dict()['layer1.weight']:
                state_dict['layer1.weight'] = state_dict['layer1.weight'][..., None, None]
            if state_dict['layer2.weight'].shape != model.state_dict()['layer2.weight']:
                state_dict['layer2.weight'] = state_dict['layer2.weight'][..., None, None]
            if state_dict['layer3.weight'].shape != model.state_dict()['layer3.weight']:
                state_dict['layer3.weight'] = state_dict['layer3.weight'][..., None, None]

        model.load_state_dict(state_dict)
        
        print(f"Load weight success: {kwargs['pth']}")
    except Exception as e:
        print(f"Load weight failed: {e}")
        return

    model.eval()
    # 用 NHWC 的 dummy
    
    dummy_input = torch.randn(*input_shape, device=device)
    
    print("Model structure:", model)
    print("Dummy input shape:", dummy_input.shape)

    # 前向测试
    try:
        with torch.no_grad():
            test_output = model(dummy_input)
        print("PyTorch output shape:", tuple(test_output.shape))
    except Exception as e:
        print(f"Inference failed: {e}")
        return

    
    try:
        dynamic_axes = {
                'input':  {0: 'N', 2: 'H', 3: 'W'},   # NHWC
                'output': {0: 'N', 2: 'H', 3: 'W'}
            } if kwargs['use_dynamic_axes'] else None
        torch.onnx.export(
            model,
            dummy_input,
            kwargs['output'],
            opset_version=11,                 
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True
        )
        print(f"ONNX exported to: {kwargs['output']}")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return


# FP16 export
@cli.command()
@click.option('--pth', type=click.Path(exists=True), default='DFAOITModel.pth', help='PyTorch 权重路径（FP32）')
@click.option('--output', default='DFAOITModel_fp16.onnx', show_default=True, type=str, help='FP16 ONNX 输出路径')
@click.option('--dummy_h', default=16, show_default=True, type=int, help='导出用占位高度（仅构图用，实际推理支持动态）')
@click.option('--dummy_w', default=16, show_default=True, type=int, help='导出用占位宽度（仅构图用，实际推理支持动态）')
@click.option("--model_arch", type=click.Choice(["DFAOITNet", "DFAOITNetConv"]), default="DFAOITNet", help="选择模型结构")
@click.option('--use_dynamic_axes', is_flag=True, default=False, help="允许运行时动态分辨率")
def export_fp16(**kwargs):
    """
    export FP16 ONNX and save FP16 .pth
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # 1.mdoel input_shape
   
    model_class = None
    if kwargs["model_arch"] == "DFAOITNet":
        model_class = DFAOITNet
        input_shape = (1, kwargs['dummy_h'], kwargs['dummy_w'], 10)   # NHWC
    elif kwargs["model_arch"] == "DFAOITNetConv":
        model_class = DFAOITNetConv
        input_shape = (1, 10, kwargs['dummy_h'], kwargs['dummy_w'])   # NCHW
    else:
        raise NotImplementedError("Model architecture '%s' is not supported" % kwargs["model_arch"])

    model = model_class().to(device)

    
    # 2. load FP32 weights
  
    try:
        state_dict = torch.load(kwargs['pth'], map_location=device)

       
        if kwargs["model_arch"] == "DFAOITNetConv":
            if state_dict['layer1.weight'].shape != model.state_dict()['layer1.weight'].shape:
                state_dict['layer1.weight'] = state_dict['layer1.weight'][..., None, None]
            if state_dict['layer2.weight'].shape != model.state_dict()['layer2.weight'].shape:
                state_dict['layer2.weight'] = state_dict['layer2.weight'][..., None, None]
            if state_dict['layer3.weight'].shape != model.state_dict()['layer3.weight'].shape:
                state_dict['layer3.weight'] = state_dict['layer3.weight'][..., None, None]

        model.load_state_dict(state_dict)
        print(f"[export_fp16] Load FP32 weight success: {kwargs['pth']}")
    except Exception as e:
        print(f"[export_fp16] Load weight failed: {e}")
        return

    
    # 3. cover and save FP16 pth
    
    model.eval()
    model.half()  # all weights -> FP16

    fp16_ckpt = os.path.splitext(kwargs['pth'])[0] + "_fp16.pth"
    torch.save(model.state_dict(), fp16_ckpt)
    print(f"[export_fp16] FP16 checkpoint saved to: {fp16_ckpt}")

    
    # 4.  FP16 dummy input
    
    dummy_input = torch.randn(*input_shape, device=device, dtype=torch.float16)
    print("Model structure:", model)
    print("Dummy input shape:", dummy_input.shape)

    
    # 5. 前向测试（FP16）
    
    try:
        with torch.no_grad():
            test_output = model(dummy_input)
        print("[export_fp16] FP16 forward OK, output shape:", tuple(test_output.shape))
    except Exception as e:
        print(f"[export_fp16] FP16 inference failed: {e}")
        return

    
    # 6. 动态输入设置
   
    try:
        dynamic_axes = {
            'input':  {0: 'N', 2: 'H', 3: 'W'},
            'output': {0: 'N', 2: 'H', 3: 'W'}
        } if kwargs['use_dynamic_axes'] else None

        
    # 7. export to FP16 ONNX
       
        torch.onnx.export(
            model,
            dummy_input,
            kwargs['output'],
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True
        )
        print(f"[export_fp16] FP16 ONNX exported to: {kwargs['output']}")
    except Exception as e:
        print(f"[export_fp16] FP16 ONNX export failed: {e}")
        return
    
# Compare FP16 & FP32 

'Comparing the output differences of the same model at FP32 and FP16 precision helps determine whether FP16 can safely replace FP32. (Random input...)'

@cli.command()
@click.option('--pth', type=click.Path(exists=True), default='DFAOITModel.pth',
              help='PyTorch 权重路径（FP32）')
@click.option('--dummy_h', default=16, show_default=True, type=int,
              help='占位高度（当未提供 input_npy 时使用）')
@click.option('--dummy_w', default=16, show_default=True, type=int,
              help='占位宽度（当未提供 input_npy 时使用）')
@click.option("--model_arch", type=click.Choice(["DFAOITNet", "DFAOITNetConv","default_fp32"]),
              default="DFAOITNet", help="选择模型结构")
@click.option('--input_npy', type=click.Path(exists=True), default=None,
              help='可选：输入特征的 .npy 文件路径（如 [1,H,W,10] 或 [1,10,H,W]）')
def compare_fp16(pth, dummy_h, dummy_w, model_arch, input_npy):
    """
    Compare the output differences between the FP32 and FP16 versions of the same model:
        – Load the same FP32 checkpoint and construct two instances of the model (FP32 and FP16).
        – Run both models on the same input (either a random tensor or a provided .npy file).
        – Compute the metrics: MAE, Max Absolute Difference, MSE, and PSNR.
    """
    import numpy as np
    import math

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 选择模型结构 & 输入形状
    if model_arch == "DFAOITNet":
        model_class = DFAOITNet
        input_shape = (1, dummy_h, dummy_w, 10)   # NHWC
    elif model_arch == "DFAOITNetConv":
        model_class = DFAOITNetConv
        input_shape = (1, 10, dummy_h, dummy_w)   # NCHW
    else:
        raise NotImplementedError("Model architecture '%s' is not supported" % model_arch)

    # 2. 构建 FP32 模型并加载权重
    model_fp32 = model_class().to(device)
    try:
        state_dict = torch.load(pth, map_location=device)

        if model_arch == "DFAOITNetConv":
            
            if state_dict['layer1.weight'].shape != model_fp32.state_dict()['layer1.weight'].shape:
                state_dict['layer1.weight'] = state_dict['layer1.weight'][..., None, None]
            if state_dict['layer2.weight'].shape != model_fp32.state_dict()['layer2.weight'].shape:
                state_dict['layer2.weight'] = state_dict['layer2.weight'][..., None, None]
            if state_dict['layer3.weight'].shape != model_fp32.state_dict()['layer3.weight'].shape:
                state_dict['layer3.weight'] = state_dict['layer3.weight'][..., None, None]

        model_fp32.load_state_dict(state_dict)
        print(f"[compare_fp16] Load FP32 weight success: {pth}")
    except Exception as e:
        print(f"[compare_fp16] Load weight failed: {e}")
        return

    model_fp32.eval()

    # 3. 构建 FP16 模型：同一权重，再 half()
    model_fp16 = model_class().to(device)
    model_fp16.load_state_dict(state_dict)  
    model_fp16.eval()
    model_fp16.half()  # 所有参数 -> FP16

    # 4. 准备输入：优先使用 input_npy，否则用随机输入
    if input_npy is not None:
        x_np = np.load(input_npy).astype(np.float32)
        print(f"[compare_fp16] Loaded input from {input_npy}, shape={x_np.shape}, dtype={x_np.dtype}")
        x = torch.from_numpy(x_np).to(device)
    else:
        x = torch.randn(*input_shape, device=device, dtype=torch.float32)
        print(f"[compare_fp16] Using random input, shape={x.shape}, dtype={x.dtype}")

    # 5. 前向推理：FP32 & FP16
    try:
        with torch.no_grad():
            y32 = model_fp32(x)              # FP32
            y16 = model_fp16(x.half())       # 同一输入，转换为 half
    except Exception as e:
        print(f"[compare_fp16] Forward inference failed: {e}")
        return

    # 6. 转为 numpy 做误差统计
    y32_np = y32.detach().cpu().numpy().astype(np.float32)
    y16_np = y16.detach().cpu().numpy().astype(np.float32)

    if y32_np.shape != y16_np.shape:
        print(f"[compare_fp16] Shape mismatch: y32 {y32_np.shape} vs y16 {y16_np.shape}")
        return

    diff = y32_np - y16_np
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    mse = float(np.mean(diff ** 2))

    # if output is [0,1],peak=1.0 ；if 0~255， peak = 255
    peak = 1.0
    #peak = 255.0
    psnr = 10.0 * math.log10((peak ** 2) / (mse + 1e-12))

    
    print("Output shape:", y32_np.shape)
    print(f"MAE:          {mae:.8f}")
    print(f"Max Abs Diff: {max_abs:.8f}")
    print(f"MSE:          {mse:.10f}")
    print(f"PSNR:         {psnr:.2f} dB")
   



# @cli.command()
# @click.option('--input', required=True, type=click.Path(exists=True), help='Input ONNX model path')
# @click.option('--height', default=1080, show_default=True, type=int, help='Target height')
# @click.option('--width', default=1920, show_default=True, type=int, help='Target width')
# @click.option('--output', required=True, type=str, help='Output ONNX model path')
# def reshape(input, height, width, output):
#     """
#     Reshape ONNX model input/output dimensions to fixed height and width
#     """
#     try:
#         # Load ONNX model
#         model = onnx.load(input)
#         print(f"Loaded ONNX model: {input}")
        
#         # Modify input tensor shape
#         input_tensor = model.graph.input[0]
#         dim_in = input_tensor.type.tensor_type.shape.dim
#         dim_in.clear()
#         dim_in.extend([
#             TensorShapeProto.Dimension(dim_value=1),       # batch size
#             TensorShapeProto.Dimension(dim_value=10),      # channels
#             TensorShapeProto.Dimension(dim_value=height),  # height
#             TensorShapeProto.Dimension(dim_value=width),   # width
#         ])
        
#         # Modify output tensor shape
#         output_tensor = model.graph.output[0]
#         dim_out = output_tensor.type.tensor_type.shape.dim
#         dim_out.clear()
#         dim_out.extend([
#             TensorShapeProto.Dimension(dim_value=1),       # batch size
#             TensorShapeProto.Dimension(dim_value=3),       # RGB channels
#             TensorShapeProto.Dimension(dim_value=height),  # height
#             TensorShapeProto.Dimension(dim_value=width),   # width
#         ])
        
#         # Update all input tensors (if there are multiple)
#         for input_tensor in model.graph.input:
#             shape = input_tensor.type.tensor_type.shape
#             if len(shape.dim) == 4:
#                 shape.dim[2].dim_value = height
#                 shape.dim[3].dim_value = width
#                 print(f"Updated {input_tensor.name} → [1,C,{height},{width}]")
        
#         # Save modified model
#         onnx.save(model, output)
#         print(f"Reshaped model saved to: {output}")
#         print(f"New dimensions: {height}x{width}")
        
#     except Exception as e:
#         print(f"Reshape failed: {e}")
#         return

# @cli.command()
# @click.option('--input', required=True, type=click.Path(exists=True), help='Input ONNX model path')
# @click.option('--height', default=1080, show_default=True, type=int, help='Target height')
# @click.option('--width', default=1920, show_default=True, type=int, help='Target width')
# @click.option('--output', required=True, type=str, help='Output ONNX model path')
# def reshape(input, height, width, output):
#     """
#     Reshape ONNX model input/output dimensions to fixed height and width (NHWC).
#       input:  [N, H, W, 10]
#       output: [N, H, W, 3]
#     """
#     try:
#         model = onnx.load(input)
#         print(f"Loaded ONNX model: {input}")

#         # ---- input: [1, H, W, 10] ----
#         in_tensor = model.graph.input[0]
#         dim_in = in_tensor.type.tensor_type.shape.dim
#         dim_in.clear()
#         dim_in.extend([
#             TensorShapeProto.Dimension(dim_value=1),         # N
#             TensorShapeProto.Dimension(dim_value=height),    # H
#             TensorShapeProto.Dimension(dim_value=width),     # W
#             TensorShapeProto.Dimension(dim_value=10),        # C=10
#         ])

#         # ---- output: [1, H, W, 3] ----
#         out_tensor = model.graph.output[0]
#         dim_out = out_tensor.type.tensor_type.shape.dim
#         dim_out.clear()
#         dim_out.extend([
#             TensorShapeProto.Dimension(dim_value=1),         # N
#             TensorShapeProto.Dimension(dim_value=height),    # H
#             TensorShapeProto.Dimension(dim_value=width),     # W
#             TensorShapeProto.Dimension(dim_value=3),         # C=3
#         ])

        
#         for t in model.graph.input:
#             shape = t.type.tensor_type.shape
#             if len(shape.dim) == 4:
#                 # NHWC: 0=N, 1=H, 2=W, 3=C
#                 shape.dim[1].dim_value = height
#                 shape.dim[2].dim_value = width
#                 print(f"Updated {t.name} → [N,H,W,C]=[1,{height},{width},{shape.dim[3].dim_value or '?'}]")

#         onnx.save(model, output)
#         print(f"Reshaped model saved to: {output}")
#         print(f"New NHWC dimensions: H={height}, W={width}")

#     except Exception as e:
#         print(f"Reshape failed: {e}")
#         return
@cli.command()
@click.option('--input', required=True, type=click.Path(exists=True), help='Input ONNX model path')
@click.option('--height', default=1080, show_default=True, type=int, help='Target height')
@click.option('--width', default=1920, show_default=True, type=int, help='Target width')
@click.option('--output', required=True, type=str, help='Output ONNX model path')
def reshape(input, height, width, output):
    """
    Reshape ONNX model input/output dimensions to fixed height and width (NCHW).
      input:  [N, 10, H, W]
      output: [N,  3, H, W]
    """
    try:
        model = onnx.load(input)
        print(f"Loaded ONNX model: {input}")

        # ---- input: [1, 10, H, W] ----
        in_tensor = model.graph.input[0]
        dim_in = in_tensor.type.tensor_type.shape.dim
        dim_in.clear()
        dim_in.extend([
            TensorShapeProto.Dimension(dim_value=1),         # N
            TensorShapeProto.Dimension(dim_value=10),        # C=10
            TensorShapeProto.Dimension(dim_value=height),    # H
            TensorShapeProto.Dimension(dim_value=width),     # W
        ])

        # ---- output: [1, 3, H, W] ----
        out_tensor = model.graph.output[0]
        dim_out = out_tensor.type.tensor_type.shape.dim
        dim_out.clear()
        dim_out.extend([
            TensorShapeProto.Dimension(dim_value=1),         # N
            TensorShapeProto.Dimension(dim_value=3),         # C=3
            TensorShapeProto.Dimension(dim_value=height),    # H
            TensorShapeProto.Dimension(dim_value=width),     # W
        ])

        # 如果有多个 input，一并改成 NCHW 的 H、W
        for t in model.graph.input:
            shape = t.type.tensor_type.shape
            if len(shape.dim) == 4:
                # NCHW: 0=N, 1=C, 2=H, 3=W
                shape.dim[2].dim_value = height
                shape.dim[3].dim_value = width
                print(
                    f"Updated {t.name} → [N,C,H,W]=["
                    f"{shape.dim[0].dim_value or 1},"
                    f"{shape.dim[1].dim_value or '?'},"
                    f"{height},{width}]"
                )

        onnx.save(model, output)
        print(f"Reshaped model saved to: {output}")
        print(f"New NCHW dimensions: H={height}, W={width}")

    except Exception as e:
        print(f"Reshape failed: {e}")
        return

@cli.command("create_default_ckpt")
@click.option("--ckpt_path", type=click.Path(exists=False,dir_okay=False,writable=True), required=True, help="file name of the output checkpoint file")
def create_default_ckpt(**kwargs):
    default_layer1_weights = [[-1.1959514617919922,0.008686563931405544,-0.1937752366065979,1.0385195016860962,0.19693408906459808,0.2250373661518097,0.6747682690620422,-1.4266048669815063,0.3059764504432678,0.399554044008255,0.27530235052108765,0.5219589471817017,0.08253791928291321,-1.228529930114746,-0.002411186695098877,-1.5230942964553833,0.9490935802459717,-0.5497636198997498,-0.5618413090705872,-0.0033106456976383924,0.561732292175293,-0.12235775589942932,0.8593060970306396,0.19862844049930573,-0.7808619737625122,0.14008311927318573,-0.019419029355049133,0.07229571789503098,0.20560601353645325,0.5272841453552246,-0.453464150428772,0.20026898384094238],[-0.32112884521484375,-0.18358077108860016,1.9115184545516968,-0.14861445128917694,0.4587515890598297,-0.24115443229675293,-0.019700361415743828,1.417148232460022,-0.344126433134079,-1.6651922464370728,-2.0527474880218506,-0.3778584599494934,-0.19501537084579468,-1.1878787279129028,0.2124522626399994,0.3017037808895111,0.2961932122707367,0.4879889488220215,-0.5376386642456055,-0.31061938405036926,-1.1316797733306885,-0.909258246421814,-0.2944614589214325,0.09972604364156723,-12.067819595336914,1.5719071626663208,-0.3163011372089386,0.6028306484222412,-0.0460175946354866,-3.6968750953674316,0.43281474709510803,0.7987827658653259],[-0.747306227684021,-0.2081216424703598,1.6365711688995361,-0.4465257525444031,0.33296605944633484,-0.29781994223594666,-0.1783263087272644,0.25825634598731995,0.19828782975673676,0.6044086813926697,0.28318580985069275,-1.072197437286377,0.008227660320699215,-0.1253279745578766,-0.0783214271068573,0.9278473854064941,-0.8836387395858765,-0.33477675914764404,-0.3048165440559387,-0.18421262502670288,-0.43830668926239014,-1.6739978790283203,-1.9730682373046875,-2.1032328605651855,-18.26808738708496,0.30352431535720825,-0.06819019466638565,0.12744948267936707,-3.205310583114624,-0.8645298480987549,-2.3543710708618164,-0.18044570088386536],[0.8616771101951599,0.3256019055843353,-0.7677901387214661,0.6860630512237549,-2.311643123626709,0.8824611902236938,-0.4228302538394928,-0.055094894021749496,0.37132036685943604,0.15426534414291382,0.7204028964042664,0.00037511371192522347,0.06806614249944687,1.2826141119003296,-0.4323919713497162,0.4851935803890228,-0.5826048851013184,0.07026584446430206,1.0444015264511108,1.5576900243759155,0.7251351475715637,0.6116061806678772,0.797327995300293,1.0446865558624268,-9.101639747619629,-2.3920469284057617,0.03743177279829979,-0.843485951423645,-0.7569155097007751,1.1005421876907349,2.3064351081848145,-0.2463560700416565],[-0.32280752062797546,0.020834308117628098,-1.4779772758483887,0.038507129997015,-0.0862230435013771,-0.24627602100372314,0.3041973114013672,0.14891831576824188,0.21034608781337738,0.44557350873947144,0.2671230435371399,0.12094470858573914,0.8080748319625854,-0.12898804247379303,-0.09779750555753708,-0.16937418282032013,-0.22657562792301178,0.611360490322113,0.20182126760482788,-0.04682571440935135,-0.34075403213500977,0.40388306975364685,0.30762118101119995,0.10296429693698883,0.15936221182346344,-0.010127464309334755,0.015392914414405823,-0.07913041114807129,-0.054541509598493576,0.34538736939430237,-0.25479599833488464,0.6687596440315247],[-0.09726037085056305,0.098298579454422,-1.1525148153305054,-0.006606179289519787,-0.2517842948436737,-0.24290618300437927,0.6350741386413574,-0.15681207180023193,0.5440434217453003,-0.24486859142780304,-0.1493503749370575,-0.006185303907841444,0.962202250957489,0.01748655177652836,0.033148352056741714,0.1075744777917862,0.06570865958929062,0.35085809230804443,0.8330182433128357,0.03134474903345108,0.6706594228744507,0.8656180500984192,-0.25917497277259827,-0.015072023496031761,0.014841337688267231,0.08644524216651917,-0.03954117372632027,0.002060219645500183,-0.9966229200363159,-0.20114032924175262,0.05233042687177658,-0.35363703966140747],[-1.1465520858764648,-0.09708844125270844,-0.2554410398006439,-0.157853364944458,0.501046359539032,-11.565048217773438,0.016360998153686523,-0.037242770195007324,-0.19894124567508698,-0.299430251121521,-0.19109228253364563,-0.13984794914722443,-0.020829396322369576,0.0602877214550972,0.08991197496652603,0.029336605221033096,0.07054608315229416,-0.17844608426094055,0.06280391663312912,-0.046232156455516815,-0.3195935785770416,-0.24996818602085114,-0.09213313460350037,-0.12648040056228638,0.7795014381408691,-0.19904878735542297,0.05011957511305809,0.07649651169776917,1.360342025756836,-0.1383044570684433,0.41980209946632385,-0.12338314205408096],[0.42042773962020874,-1.577932357788086,1.0078223943710327,-1.0605205297470093,0.35287901759147644,0.9932248592376709,1.653694748878479,-0.8026828765869141,-0.7746771574020386,0.06780596822500229,-4.55473518371582,0.45145905017852783,0.6953707337379456,1.0300620794296265,-3.5986011028289795,0.5053223967552185,-0.32716161012649536,-2.3369805812835693,1.6742128133773804,-0.012954497709870338,1.4992144107818604,-0.4427145719528198,0.5302993059158325,0.009094552136957645,-0.3936215937137604,0.13822458684444427,-0.20050887763500214,0.7013229131698608,0.507688581943512,-11.029073715209961,0.09855765849351883,1.7905081510543823],[0.09672455489635468,-0.4092945158481598,1.5038893222808838,-1.0378752946853638,0.5929093956947327,-0.1868642121553421,0.7708125114440918,-0.09402211755514145,3.138631582260132,0.29342177510261536,0.5072694420814514,-7.3871684074401855,-0.7743351459503174,2.616819381713867,0.5649107694625854,-0.5819246768951416,0.8473128080368042,0.8983940482139587,-0.8343707919120789,-0.44072848558425903,-1.1694471836090088,-0.49466848373413086,1.1135810613632202,-3.7073304653167725,-0.338094562292099,0.25775885581970215,-0.36888587474823,-3.449357271194458,-18.542001724243164,2.04738712310791,-1.3176251649856567,1.6326231956481934],[1.7182526588439941,0.2149658501148224,-1.0617996454238892,1.1388027667999268,-7.008763313293457,-4.764480113983154,-2.8290886878967285,0.07904283702373505,1.0919829607009888,-4.736726760864258,0.7009994983673096,0.6952314972877502,-0.3030679225921631,2.4487380981445312,0.10241623222827911,-0.3376392126083374,2.523329973220825,0.8281531929969788,2.6658360958099365,3.98511004447937,0.5961707234382629,0.8543791770935059,0.11942192912101746,-0.1556708961725235,-1.1582932472229004,-3.1259171962738037,-0.2020474076271057,0.061312925070524216,-1.456930160522461,-1.5318260192871094,1.546918272972107,1.4512051343917847]]
    default_layer1_bias = [0.94788074, 1.1071887, 0.17212379, 0.96233034, 0.41116345, 0.29424584, 0.2524292, 0.56384075, 0.17144501, 0.70693594, 0.60297567, 0.3995347, -0.11675993, 0.6551702, 0.8530541, 0.8074819, -0.14349526, -0.02273662, 0.22370361, -0.15556079, -0.10795841, -0.04227992, -0.2230586 , 0.57490826, 0.35543177, 0.30263418, -0.01075008, 0.57908285, 0.35843596, 0.50211096, 0.17400633, 0.00989959]

    default_layer2_weights = [[0.4855324625968933,0.4931071400642395,0.10380104184150696,1.3709137439727783,-0.32564836740493774,-1.9942959547042847,1.386703610420227,0.3648749589920044,2.0565879344940186,0.02649473398923874,-0.682405412197113,0.19458186626434326,-0.17608760297298431,0.7249542474746704,0.38143762946128845,-0.060828015208244324],[-0.15595318377017975,-0.7519590258598328,0.07108820974826813,0.7860511541366577,-0.8302670121192932,-0.22049321234226227,0.14070487022399902,0.2799070477485657,-2.1001341342926025,-0.057683296501636505,-0.014779577031731606,-0.7197736501693726,0.49455758929252625,-0.10179200768470764,1.1618152856826782,-0.24636442959308624],[0.3151477575302124,0.24846771359443665,0.10003579407930374,0.4288254976272583,-0.13821683824062347,-0.38535216450691223,0.4654299318790436,0.02280261740088463,1.0234557390213013,0.020582543686032295,0.15840017795562744,0.10749263316392899,1.3036932945251465,1.1788554191589355,0.36588141322135925,-0.31320664286613464],[-0.5017208456993103,0.9030942916870117,0.7830336093902588,-0.25512924790382385,1.2962630987167358,0.9618062973022461,1.457244873046875,-0.8943830728530884,0.32682719826698303,0.18003877997398376,-0.39415791630744934,0.5483765006065369,0.36082711815834045,-0.04673543572425842,0.7956789135932922,-0.03845515102148056],[0.5876906514167786,0.0006338664679788053,1.0915290117263794,0.6103436946868896,0.6506611108779907,0.5670680999755859,-0.7777470946311951,-0.16347284615039825,0.43014267086982727,-0.2577018439769745,-0.4195305109024048,-0.30550628900527954,-0.050949689000844955,-1.3560826778411865,0.2832963466644287,-0.38394439220428467],[-1.0235261917114258,-0.17534871399402618,-4.8392109870910645,0.16609767079353333,0.823218584060669,-3.0942225456237793,1.2025632858276367,4.317955493927002,1.4691383838653564,0.16346687078475952,-0.4719548523426056,0.5866063833236694,0.4686736762523651,0.7369234561920166,0.008164627477526665,-0.17022749781608582],[0.5724796056747437,-0.03635172173380852,-0.1545363813638687,1.039222240447998,-0.3984861969947815,0.0296646561473608,-0.2836281359195709,1.1882721185684204,0.1366232931613922,-0.204912930727005,0.007424644660204649,-1.0028709173202515,-1.202796220779419,-0.05829732492566109,-1.7417320013046265,0.04341378062963486],[0.23129233717918396,-1.3527560234069824,-0.42133307456970215,0.12211021035909653,-0.6104382872581482,-0.1868549883365631,0.40106314420700073,-0.4845059812068939,-0.0197924692183733,0.12606611847877502,0.5827474594116211,0.6700010299682617,0.05103679001331329,0.5389978885650635,-0.23360837996006012,0.23208793997764587],[0.67277991771698,-1.7657382488250732,0.02171258069574833,0.08498626202344894,0.28049126267433167,-0.7595258355140686,0.2776927053928375,-0.35044965147972107,0.4565054476261139,-0.15233522653579712,0.4872066378593445,1.300654649734497,-1.8504332304000854,0.8463921546936035,-0.2597326338291168,0.22363460063934326],[0.010712376795709133,-0.4714129567146301,0.35520726442337036,1.446079969406128,0.5455968976020813,-1.3260340690612793,0.19403614103794098,0.049743350595235825,0.47353142499923706,0.12994804978370667,-0.9549532532691956,-0.015963705256581306,-0.38273322582244873,-0.9555284976959229,-0.19458109140396118,-0.36547940969467163],[-0.7875256538391113,0.3425721824169159,-1.2423834800720215,-0.4068931043148041,1.01790189743042,0.22240284085273743,0.3558140993118286,0.2660442292690277,-0.5474269390106201,-0.3615276515483856,1.0313496589660645,0.4277484714984894,-0.45575040578842163,2.0379459857940674,0.7719808220863342,-0.2623959183692932],[-0.11566110700368881,0.15549589693546295,0.8930090665817261,0.9349038600921631,0.7597079873085022,0.05463235452771187,0.8461006879806519,0.12101807445287704,-0.6079994440078735,0.004490119405090809,1.1005821228027344,-0.27633994817733765,0.14224164187908173,0.34271568059921265,0.16557490825653076,-0.3457490801811218],[0.39591705799102783,-0.2504904568195343,-0.031493984162807465,-0.18310216069221497,0.7079054117202759,-0.1883230060338974,0.34667375683784485,-0.07623184472322464,0.2013239711523056,-0.11317132413387299,-0.11856567114591599,-0.015944575890898705,3.6823577880859375,0.18516729772090912,2.315159797668457,-0.2989019751548767],[0.5496065020561218,0.7007465958595276,0.17294569313526154,-0.7926169037818909,0.6532262563705444,1.5787855386734009,0.2006942331790924,-0.18958544731140137,-0.28471508622169495,-0.18025937676429749,0.378176212310791,-0.3235878646373749,2.6459484100341797,-0.8277072310447693,0.23949141800403595,0.18704336881637573],[-0.3619232475757599,-0.583285391330719,-0.9040329456329346,-0.4538056254386902,1.512380838394165,0.31583330035209656,0.9610786437988281,-0.08723882585763931,1.027336597442627,0.01928548514842987,-0.2750968933105469,-0.46684974431991577,0.1602950543165207,-1.5791192054748535,0.5341970920562744,-0.39536944031715393],[0.1799522489309311,0.923843264579773,0.21015281975269318,-0.2897454500198364,0.39070063829421997,0.41485702991485596,-0.31079035997390747,-0.0005115203093737364,-0.5140342116355896,-0.1640966832637787,1.1543800830841064,-0.518275260925293,-1.577041506767273,-0.013632168993353844,-0.4515509009361267,0.06432336568832397],[0.09957201778888702,-0.09353253990411758,0.12263648957014084,-0.690255880355835,-0.19034689664840698,0.8746852874755859,-0.9009305834770203,-0.313703328371048,0.5521869659423828,-0.19129839539527893,1.2473962306976318,0.061218325048685074,0.08761314302682877,0.654431164264679,-0.14619199931621552,-0.33451715111732483],[0.07974260300397873,-0.37096571922302246,-0.06174737960100174,-0.8915731906890869,-0.5108177661895752,1.1457362174987793,-0.493486613035202,0.07922235876321793,0.22418564558029175,0.19242951273918152,0.3291654884815216,0.38492754101753235,-0.8371371626853943,-0.4161410927772522,-0.8019778728485107,0.21948641538619995],[-0.7994012236595154,-1.8920094966888428,0.6720749139785767,-0.17469771206378937,-0.3729900121688843,0.023695070296525955,-0.021520232781767845,-0.026918785646557808,0.44524917006492615,-0.1861088126897812,0.978980302810669,-0.5237755179405212,-2.601365804672241,-0.5347180962562561,-1.005389928817749,-0.08774088323116302],[0.6610136032104492,-0.33315953612327576,-0.5808750987052917,1.0279815196990967,-0.6386284828186035,-1.6286004781723022,-0.024559464305639267,1.7753922939300537,-0.1544920802116394,-0.18138431012630463,-0.347635954618454,-0.8881171941757202,-0.17921799421310425,1.1944081783294678,0.06857842206954956,-0.12952955067157745],[-0.3381940424442291,-1.0793044567108154,0.37167641520500183,0.32332950830459595,0.008518817834556103,0.2817102074623108,-0.4100678265094757,-0.21096022427082062,-0.9516218900680542,-0.17738547921180725,0.28829070925712585,0.32393237948417664,0.14444445073604584,-0.7172508239746094,-0.06652387976646423,0.13653846085071564],[-0.4724299907684326,3.959125280380249,-0.595298171043396,-0.11153925955295563,0.05432121083140373,0.05404157564043999,0.31248193979263306,-0.45154955983161926,-0.5873998403549194,0.32329025864601135,-1.105285406112671,0.47186627984046936,-0.7073745727539062,0.04058119282126427,-0.25135353207588196,0.06204336881637573],[-0.09862303733825684,-0.16838370263576508,-0.6563517451286316,-0.33635249733924866,0.7841166257858276,-0.2100493609905243,0.23235458135604858,0.6351258158683777,0.25170159339904785,-0.08564119040966034,1.1780649423599243,-0.8901026248931885,-0.3950941264629364,0.43442872166633606,-0.34140560030937195,-0.20018291473388672],[-0.9007940292358398,-0.7312206029891968,-1.2381744384765625,0.3201162815093994,0.49314022064208984,-0.41883111000061035,1.3737378120422363,-0.07130175083875656,0.004479328170418739,-0.10741174221038818,-0.40654563903808594,0.28353869915008545,0.11062107980251312,-0.11100152879953384,-0.13095390796661377,-0.17056450247764587],[1.441879153251648,0.4924049377441406,3.0137619972229004,1.6133272647857666,-0.667236328125,0.12797600030899048,-0.5594787001609802,1.2254446744918823,-0.20554421842098236,-0.18752239644527435,-0.02557636797428131,1.4591253995895386,-1.95885169506073,1.8351293802261353,1.1069278717041016,-0.23766392469406128],[0.1360471546649933,0.2355700135231018,0.1110483780503273,0.8886839151382446,0.15384434163570404,-0.2800663113594055,-0.3006570041179657,0.699007511138916,-0.4770301878452301,0.0030932750087231398,0.3729117512702942,0.5146426558494568,-0.08564124256372452,1.5841659307479858,-0.23869852721691132,-0.17625218629837036],[0.06073814630508423,0.3504723012447357,0.20597414672374725,0.1500907689332962,-0.2402230054140091,0.31028828024864197,-0.2294100522994995,-0.22374005615711212,-0.3091212511062622,0.1410476118326187,0.028931617736816406,0.2358057200908661,0.10067448019981384,0.09879446774721146,0.27338817715644836,0.21246185898780823],[-0.24570997059345245,-0.02542683482170105,0.285625159740448,0.7275424003601074,-0.18951316177845,-0.2462969720363617,0.04082686826586723,-0.5502161383628845,0.813132107257843,0.06461475044488907,0.1457694172859192,0.4810294210910797,-0.5208631753921509,1.0722259283065796,-1.654968500137329,0.1312693953514099],[0.4604164958000183,-0.2767646908760071,6.574335098266602,4.611900806427002,-0.5481435060501099,0.12950026988983154,1.8382775783538818,0.19294202327728271,0.26653245091438293,-0.30541419982910156,0.9513653516769409,0.7260265946388245,-0.18213582038879395,-0.2617158889770508,0.21246477961540222,-0.10989498347043991],[1.8427926301956177,0.3394126892089844,0.9705246090888977,-0.2986394762992859,1.8211095333099365,1.890419840812683,1.187295913696289,0.15032079815864563,-0.35558509826660156,0.038662560284137726,1.474493384361267,0.34445011615753174,0.5976400375366211,-1.4448442459106445,0.4462164342403412,0.210927814245224],[-0.06704511493444443,0.8425297737121582,0.7273149490356445,0.19566695392131805,-0.04238157719373703,0.1792774349451065,-0.8822489380836487,-0.8191877007484436,1.3391218185424805,-0.31876859068870544,1.0471078157424927,0.42806822061538696,0.02055804617702961,0.8125109672546387,-1.0384373664855957,0.06425516307353973],[-0.29669493436813354,0.33801916241645813,0.5476243495941162,1.0462514162063599,-0.7626770734786987,-0.11118371039628983,-0.3406493067741394,-1.163106918334961,-0.6331011056900024,0.0976642593741417,0.656583845615387,1.0642417669296265,-0.5848557353019714,0.6343390345573425,-0.9152898192405701,-0.2263033539056778]]
    default_layer2_bias = [1.0024987, 0.7019234, -0.31018218, -0.2966683, 0.57583773, 0.38118008, 0.03561725, 1.030341, 0.00273793, -0.092371, -0.50508595, -0.23851559, 1.160991, -0.05685527, 0.00209139, -0.08914163]

    default_layer3_weights = [[-0.1631498485803604,0.5885962843894958,-0.45753711462020874],[-0.29011666774749756,-1.7414355278015137,-0.254714697599411],[-0.3687281906604767,-0.42367491126060486,0.4668368399143219],[0.5017827749252319,-0.35678642988204956,-0.7905645966529846],[-0.5253183841705322,0.17899642884731293,-0.08671782165765762],[0.05534752458333969,-0.5685427784919739,-0.6783115863800049],[-0.24065136909484863,-0.5730992555618286,0.0858161449432373],[-0.13739177584648132,-0.06401057541370392,-0.6701674461364746],[-0.17737291753292084,-0.17955224215984344,-0.19217659533023834],[0.4204261302947998,0.5289128422737122,0.3415016531944275],[0.4365029036998749,0.2399706393480301,0.4704277217388153],[-0.1925477236509323,0.21371600031852722,0.3034309446811676],[-1.1344658136367798,-0.982851505279541,-0.9596925377845764],[-0.32806965708732605,-0.21451649069786072,-0.20496052503585815],[-0.43247565627098083,-0.06669460237026215,-0.03655197471380234],[0.06050315126776695,0.32564544677734375,0.19158335030078888]]
    default_layer3_bias = [-0.7872798 ,  0.04166554, -0.61643946]

    model = DFAOITNet()

    load_existing_weights(model, default_layer1_weights, default_layer1_bias, default_layer2_weights, default_layer2_bias, default_layer3_weights, default_layer3_bias)
    
    os.makedirs(os.path.dirname(kwargs["ckpt_path"]), exist_ok=True)

    torch.save(model.state_dict(), kwargs["ckpt_path"])


if __name__ == '__main__':
    cli()
