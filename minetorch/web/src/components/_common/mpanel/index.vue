<template>
  <div class="mpanel">
    <div class="mpanel-heading">
      <span class="pull-left">{{ title }}</span>
      <slot name="tip"></slot>
    </div>
    <div class="mpanel-body">
      <div class="mpanel-body-wrapper">
        <slot></slot>
      </div>
    </div>
  </div>
</template>
<script>
export default {
  props: {
    title: {
      type: Object,
      required: true
    }
  },
  data () {
    return {}
  }
}
</script>
<style lang="scss">
.mpanel {
  background-color: #fff;
  margin-bottom: 15px;
  border: 1px solid #ddd;
  min-width: 1040px;
  .mpanel-heading {
    height: 40px;
    line-height: 40px;
    padding-left: 20px;
    border-bottom: 1px solid #ddd;
    font-size: 14px;
    font-weight: bold;
    color: #333;
    background-color: #fafafc;
  }
  .mpanel-body {
    padding: 0 20px;
    .mpanel-body-wrapper {
      padding: 20px 0;
    }
  }
  .mpanel {
    min-width: auto;
  }
}
</style>
