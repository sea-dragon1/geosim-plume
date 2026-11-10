# 自动生成的 Makefile
CC = icx
MKL_INTERFACE = lp64
CFLAGS = -O2 -qopenmp -mcmodel=medium
LDFLAGS = -qmkl -lpthread -lm -Wl,--no-relax
#CFLAGS = -O3 -xCORE-AVX512 -mcmodel=medium
#LDFLAGS =  -qmkl -lpthread -qopt-zmm-usage=high

TARGETS = i2fast in2fast i2jslab

all: $(TARGETS)
	@echo "✅ 编译完成！"

i2fast: i2fast.c
	$(CC) $(CFLAGS) -o $@ i2fast.c $(LDFLAGS)

in2fast: in2fast.c
	$(CC) $(CFLAGS) -o $@ in2fast.c  $(LDFLAGS)

i2jslab: i2jslab.c
	$(CC) $(CFLAGS) -o $@ i2jslab.c $(LDFLAGS)

clean:
	rm -f $(TARGETS)
	@echo "🧹 已清除所有可执行文件"
