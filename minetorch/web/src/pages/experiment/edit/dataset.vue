<template>
  <div class="experiment-dataset">
    <pre>
      <code>
        1. 请求 /api/$components (这里是datasets) 接口获取所有可选择的 $components
        2. 用户选择某个 component 后，弹窗显示配置表单
        3. 表单被正确提交后，进到下一个 Component 的步骤
        4. 如果用户回退到当前 Component 的配置，显示用户已经配置好信息，然后底部两个按钮「修改」「选择其他的$resource」
           点击「修改」后，弹窗显示配置表单，保存后刷新页面信息（不要进到下一步）。点击 「选择其他$resource」后，
           重复步骤 1
      </code>
    </pre>
  </div>
</template>
<script>
import dateFormat from 'dateformat'
import moment from 'moment'

export default {
  components: {
  },
  data () {
    return {
    }
  },
  mounted () {
  },
  methods: {
  }
}
</script>
<style lang="scss">
</style>
