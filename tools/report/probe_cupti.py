"""Read-only diagnostic: distinguish CUDA execution, CUPTI tracing and legacy counter failures."""
import ctypes as c
from ctypes import wintypes
import json,os,sys
from pathlib import Path

toolkit=Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9')
handles=[os.add_dll_directory(str(toolkit/'extras/CUPTI/lib64')),os.add_dll_directory(str(toolkit/'bin'))]
driver=c.WinDLL('nvcuda.dll');cupti=c.WinDLL(str(toolkit/'extras/CUPTI/lib64/cupti64_2025.2.0.dll'))
steps=[]
def invoke(lib,name,types,values):
 fn=getattr(lib,name);fn.argtypes=types;fn.restype=c.c_int
 status=fn(*values);message=c.c_char_p()
 get=cupti.cuptiGetResultString if lib is cupti else driver.cuGetErrorString
 get.argtypes=[c.c_int,c.POINTER(c.c_char_p)];get.restype=c.c_int
 get(status,c.byref(message))
 steps.append(dict(api=name,status=status,message=message.value.decode() if message.value else None))
 return status
uint=c.c_uint32;ptr=c.c_void_p
version=uint();invoke(cupti,'cuptiGetVersion',[c.POINTER(uint)],[c.byref(version)])
invoke(driver,'cuInit',[uint],[0])
device=c.c_int();invoke(driver,'cuDeviceGet',[c.POINTER(c.c_int),c.c_int],[c.byref(device),0])
ctx=ptr();retained=invoke(driver,'cuDevicePrimaryCtxRetain',[c.POINTER(ptr),c.c_int],[c.byref(ctx),device])==0
if retained:invoke(driver,'cuCtxSetCurrent',[ptr],[ctx])
count=uint()
domain_status=invoke(cupti,'cuptiDeviceGetNumEventDomains',[c.c_int,c.POINTER(uint)],[device,c.byref(count)])
if domain_status==0 and count.value:
 size=c.c_size_t(count.value*c.sizeof(uint));domains=(uint*count.value)()
 invoke(cupti,'cuptiDeviceEnumEventDomains',[c.c_int,c.POINTER(c.c_size_t),c.POINTER(uint)],[device,c.byref(size),domains])
 for domain in list(domains)[:2]:
  events=uint();invoke(cupti,'cuptiEventDomainGetNumEvents',[uint,c.POINTER(uint)],[domain,c.byref(events)])
group=ptr()
created=invoke(cupti,'cuptiEventGroupCreate',[ptr,c.POINTER(ptr),uint],[ctx,c.byref(group),0])==0
if created:invoke(cupti,'cuptiEventGroupDestroy',[ptr],[group])
callback_type=c.WINFUNCTYPE(None,ptr,c.c_int,uint,ptr)
callback=callback_type(lambda userdata,domain,cbid,data:None);subscriber=ptr()
subscribed=invoke(cupti,'cuptiSubscribe',[c.POINTER(ptr),callback_type,ptr],[c.byref(subscriber),callback,None])==0
if subscribed:invoke(cupti,'cuptiUnsubscribe',[ptr],[subscriber])
enabled=invoke(cupti,'cuptiActivityEnable',[c.c_int],[10])==0
if enabled:invoke(cupti,'cuptiActivityDisable',[c.c_int],[10])
if retained:invoke(driver,'cuDevicePrimaryCtxRelease_v2',[c.c_int],[device])
result=dict(cupti_dll=str(toolkit/'extras/CUPTI/lib64/cupti64_2025.2.0.dll'),cupti_api_version=version.value,
 is_administrator=bool(c.windll.shell32.IsUserAnAdmin()),event_domain_count=count.value,steps=steps,
 system_settings_changed=False)
print(json.dumps(result,indent=2))
if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
