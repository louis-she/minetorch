<template>
  <div ref="chart" class="chart"></div>
</template>

<script>
export default {
  props: {
    option: {
      type: 'Object',
      default: {}
    }
  },
  data () {
    return {
      chart: null
    }
  },
  watch: {
    option (newVal) {
      this.chart.setOption(newVal)
    }
  },
  mounted () {
    this.chart = echarts.init(this.$refs.chart)
    this.chart.setOption(this.option)
  }
}
</script>

<style lang="scss">
.chart {
  width: 100%;
  height: 700px;
}
</style>
