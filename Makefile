
bin/XS.so:
	@mkdir -p bin
	nvcc --compiler-options='-fPIC' -use_fast_math -lineinfo --ptxas-options=-v -shared -o bin/XS.so src/XSMD.cu src/vdW.cu src/WaasKirf.cu

clean:
	@echo " Cleaning ...";
	@rm bin/*
